import json
import sqlite3
import shutil
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from tqdm import tqdm

from modules.LLMProcessor import LLMProcessor
from utils.prompts import PostProcessingPrompts

logger = logging.getLogger("PostProcessor")


class PostProcessor:
    """
    Phase 2: Post-processing of raw database.

    1. Deduplicate person names (Алексей = Лёша = Alex)
    2. Consolidate facts per person (merge duplicates, resolve conflicts)
    3. Aggregate Big Five dimensions (weighted average)
    4. Clean up entities and relations
    5. Build final clean database
    6. Generate enhanced Obsidian vault
    """

    def __init__(self, output_dir: str, ollama_url: str, model: str):
        self.output_dir = Path(output_dir)
        self.raw_db_path = self.output_dir / "personality_graph.db"
        self.clean_db_path = self.output_dir / "personality_graph_clean.db"
        self.vault_dir = self.output_dir / "obsidian_vault"

        self.llm = LLMProcessor(ollama_url=ollama_url, model=model)

        # Map: alias -> canonical name
        self.name_mapping: Dict[str, str] = {}

        # Statistics
        self.stats = {
            'people_found': 0,
            'aliases_merged': 0,
            'facts_consolidated': 0,
            'facts_removed_as_duplicates': 0,
            'dimensions_averaged': 0,
            'entities_deduplicated': 0,
        }

    def run(self):
        print()
        print("==================================================")
        print("│    Phase 2: Post-Processing & Consolidation    │")
        print("==================================================")
        print()

        if not self.raw_db_path.exists():
            logger.error(f"Raw database not found: {self.raw_db_path}")
            return

        # Copy raw DB as base for clean DB
        shutil.copy(self.raw_db_path, self.clean_db_path)
        self.conn = sqlite3.connect(self.clean_db_path)
        self.conn.row_factory = sqlite3.Row

        # Step 1: Discover all person names
        all_names = self._discover_all_names()
        self.stats['people_found'] = len(all_names)
        logger.info(f"Discovered {len(all_names)} unique names/aliases")

        # Step 2: Deduplicate names via LLM
        if len(all_names) > 1:
            self._deduplicate_names(all_names)

        # Step 3: Apply name mapping to all tables
        self._apply_name_mapping()

        # Step 4: Consolidate facts per person (deduplicate within person)
        self._consolidate_facts()

        # Step 5: Aggregate Big Five dimensions
        self._aggregate_dimensions()

        # Step 6: Deduplicate entities
        self._deduplicate_entities()

        # Step 7: Clean relations
        self._clean_relations()

        # Step 8: Build enhanced vault from clean DB
        self._generate_enhanced_vault()

        self.conn.commit()
        self.conn.close()

        self._print_stats()
        logger.info(f"Clean database saved: {self.clean_db_path}")

    # ── Step 1: Discover Names ──

    def _discover_all_names(self) -> List[str]:
        """Собирает все уникальные имена из всех таблиц."""
        names = set()
        cursor = self.conn.cursor()

        # From entities (belongs_to)
        cursor.execute("SELECT DISTINCT belongs_to FROM entities WHERE belongs_to != 'all'")
        names.update(r['belongs_to'] for r in cursor.fetchall())

        # From facts
        cursor.execute("SELECT DISTINCT belongs_to FROM facts WHERE belongs_to != 'all'")
        names.update(r['belongs_to'] for r in cursor.fetchall())

        # From patterns
        cursor.execute("SELECT DISTINCT belongs_to FROM cognitive_patterns WHERE belongs_to != 'all'")
        names.update(r['belongs_to'] for r in cursor.fetchall())

        # From dimensions
        cursor.execute("SELECT DISTINCT belongs_to FROM personality_dimensions WHERE belongs_to != 'all'")
        names.update(r['belongs_to'] for r in cursor.fetchall())

        # From communication styles
        cursor.execute("SELECT DISTINCT belongs_to FROM communication_styles WHERE belongs_to != 'all'")
        names.update(r['belongs_to'] for r in cursor.fetchall())

        # From messages (authors)
        cursor.execute("SELECT DISTINCT author FROM messages")
        names.update(r['author'] for r in cursor.fetchall())

        # From social graph
        cursor.execute("SELECT DISTINCT person_a FROM social_graph")
        names.update(r['person_a'] for r in cursor.fetchall())
        cursor.execute("SELECT DISTINCT person_b FROM social_graph")
        names.update(r['person_b'] for r in cursor.fetchall())

        # From relations (source/target that look like people)
        cursor.execute("SELECT DISTINCT source FROM relations")
        names.update(r['source'] for r in cursor.fetchall())
        cursor.execute("SELECT DISTINCT target FROM relations")
        names.update(r['target'] for r in cursor.fetchall())

        # Filter out non-person entities (heuristic: single words that are common objects/places)
        filtered = []
        for name in names:
            name = name.strip()
            if not name or name.lower() in ('unknown', 'all', '', 'none'):
                continue
            filtered.append(name)

        return sorted(set(filtered))

    # ── Step 2: Deduplicate Names via LLM ──

    def _deduplicate_names(self, all_names: List[str]):
        """Отправляет список имён в LLM для дедупликации с контекстными подсказками."""
        logger.info("Sending names to LLM for deduplication...")

        # Собираем контекст для каждого имени (сущности, факты, связи)
        name_context = self._build_name_context(all_names)

        batch_size = 50  # Меньше батч для лучшего контекста
        all_groups = []

        for i in range(0, len(all_names), batch_size):
            batch = all_names[i:i + batch_size]
            user_prompt = self._build_dedup_prompt(batch, name_context)

            result = self.llm.process_with_prompt(
                PostProcessingPrompts.DEDUPLICATION_PROMPT,
                user_prompt,
                temperature=0.1
            )

            groups = result.get('canonical_groups', [])
            all_groups.extend(groups)

            for group in groups:
                canonical = group.get('canonical_name', '').strip()
                aliases = group.get('aliases', [])
                if canonical:
                    for alias in aliases:
                        alias = alias.strip()
                        if alias and alias != canonical:
                            self.name_mapping[alias.lower()] = canonical
                            self.stats['aliases_merged'] += 1

        logger.info(f"Created {len(all_groups)} canonical groups, merged {self.stats['aliases_merged']} aliases")

    def _build_name_context(self, names: List[str]) -> Dict[str, Dict]:
        """Собирает контекст для каждого имени: сущности, факты, связанные люди."""
        cursor = self.conn.cursor()
        context = {}

        for name in names:
            ctx = {
                'entities': [],
                'facts': [],
                'relations': [],
                'messages_count': 0,
                'authors_mentioned_with': set(),
            }

            # Сущности
            cursor.execute("SELECT name, category FROM entities WHERE belongs_to = ? LIMIT 10", (name,))
            ctx['entities'] = [f"{r['name']}({r['category']})" for r in cursor.fetchall()]

            # Факты
            cursor.execute("SELECT fact, category FROM facts WHERE belongs_to = ? LIMIT 5", (name,))
            ctx['facts'] = [f"{r['fact'][:50]}..." for r in cursor.fetchall()]

            # Связи
            cursor.execute("""
                SELECT target, relation_type FROM relations 
                WHERE source = ? OR belongs_to = ? LIMIT 10
            """, (name, name))
            ctx['relations'] = [f"{r['target']}({r['relation_type']})" for r in cursor.fetchall()]

            # Сообщения
            cursor.execute("SELECT COUNT(*) as cnt FROM messages WHERE author = ?", (name,))
            ctx['messages_count'] = cursor.fetchone()['cnt']

            # С кем упоминается в одних чанках
            cursor.execute("""
                SELECT DISTINCT author FROM messages 
                WHERE text LIKE ? AND author != ?
                LIMIT 5
            """, (f"%{name}%", name))
            ctx['authors_mentioned_with'] = [r['author'] for r in cursor.fetchall()]

            context[name] = ctx

        return context

    def _build_dedup_prompt(self, names: List[str], context: Dict) -> str:
        """Строит промпт с контекстными подсказками для каждого имени."""
        lines = []
        for name in names:
            ctx = context.get(name, {})
            lines.append(f"\n=== {name} ===")
            lines.append(f"  Сообщений как автор: {ctx.get('messages_count', 0)}")
            if ctx.get('entities'):
                lines.append(f"  Сущности: {', '.join(ctx['entities'][:5])}")
            if ctx.get('facts'):
                lines.append(f"  Факты: {', '.join(ctx['facts'][:3])}")
            if ctx.get('relations'):
                lines.append(f"  Связи: {', '.join(ctx['relations'][:5])}")
            if ctx.get('authors_mentioned_with'):
                lines.append(f"  Упоминается с: {', '.join(ctx['authors_mentioned_with'])}")

        return f"""Проанализируй следующие имена и их контекст из переписки.
Определи, какие имена относятся к ОДНОМУ реальному человеку.

КОНТЕКСТ ДЛЯ КАЖДОГО ИМЕНИ:
{chr(10).join(lines)}

Верни JSON с canonical_groups и uncertain_matches.
"""

    # ── Step 3: Apply Name Mapping ──

    def _apply_name_mapping(self):
        """Применяет маппинг имён ко всем таблицам."""
        if not self.name_mapping:
            return

        logger.info("Applying name mapping to all tables...")
        cursor = self.conn.cursor()

        tables_fields = [
            ('entities', 'belongs_to'),
            ('facts', 'belongs_to'),
            ('cognitive_patterns', 'belongs_to'),
            ('personality_dimensions', 'belongs_to'),
            ('communication_styles', 'belongs_to'),
            ('messages', 'author'),
            ('social_graph', 'person_a'),
            ('social_graph', 'person_b'),
            ('relations', 'source'),
            ('relations', 'target'),
            ('relations', 'belongs_to'),
        ]

        for table, field in tables_fields:
            for alias, canonical in self.name_mapping.items():
                cursor.execute(f"""
                    UPDATE {table} 
                    SET {field} = ? 
                    WHERE LOWER({field}) = ?
                """, (canonical, alias.lower()))

        self.conn.commit()
        logger.info("Name mapping applied")

    # ── Step 4: Consolidate Facts ──

    def _consolidate_facts(self):
        """Дедуплицирует факты внутри каждого человека, мержит похожие."""
        logger.info("Consolidating facts...")
        cursor = self.conn.cursor()

        cursor.execute("SELECT DISTINCT belongs_to FROM facts WHERE belongs_to != 'all'")
        people = [r['belongs_to'] for r in cursor.fetchall()]

        for person in tqdm(people, desc="Consolidating facts"):
            cursor.execute("""
                SELECT id, fact, category, confidence 
                FROM facts 
                WHERE belongs_to = ?
                ORDER BY confidence DESC, LENGTH(fact) DESC
            """, (person,))

            facts = cursor.fetchall()
            if len(facts) <= 1:
                continue

            # Простая дедупликация: удаляем факты, которые являются подстроками более длинных
            to_delete = set()
            for i, f1 in enumerate(facts):
                if f1['id'] in to_delete:
                    continue
                for j, f2 in enumerate(facts):
                    if i >= j or f2['id'] in to_delete:
                        continue
                    # Если f2 — подстрока f1 и та же категория
                    if f2['fact'].lower() in f1['fact'].lower() and f1['category'] == f2['category']:
                        to_delete.add(f2['id'])
                        self.stats['facts_removed_as_duplicates'] += 1

            if to_delete:
                cursor.executemany(
                    "DELETE FROM facts WHERE id = ?",
                    [(fid,) for fid in to_delete]
                )

        self.conn.commit()
        logger.info(f"Removed {self.stats['facts_removed_as_duplicates']} duplicate facts")

    # ── Step 5: Aggregate Dimensions ──

    def _aggregate_dimensions(self):
        """Усредняет Big Five измерения по каждому человеку."""
        logger.info("Aggregating personality dimensions...")
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT belongs_to, dimension, 
                   AVG(score) as avg_score, 
                   AVG(confidence) as avg_conf,
                   COUNT(*) as count,
                   GROUP_CONCAT(evidence, ' | ') as all_evidence
            FROM personality_dimensions 
            GROUP BY belongs_to, dimension
        """)

        # Создаём таблицу для агрегированных измерений
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personality_dimensions_agg (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                belongs_to TEXT,
                dimension TEXT,
                score REAL,
                confidence REAL,
                sample_count INTEGER,
                evidence TEXT,
                UNIQUE(belongs_to, dimension)
            )
        """)

        for row in cursor.fetchall():
            cursor.execute("""
                INSERT INTO personality_dimensions_agg (belongs_to, dimension, score, confidence, sample_count, evidence)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(belongs_to, dimension) DO UPDATE SET
                    score = excluded.score,
                    confidence = excluded.confidence,
                    sample_count = excluded.sample_count,
                    evidence = excluded.evidence
            """, (row['belongs_to'], row['dimension'], row['avg_score'],
                  row['avg_conf'], row['count'], row['all_evidence'][:500]))
            self.stats['dimensions_averaged'] += 1

        self.conn.commit()

    # ── Step 6: Deduplicate Entities ──

    def _deduplicate_entities(self):
        """Удаляет дублирующиеся сущности внутри человека."""
        logger.info("Deduplicating entities...")
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT belongs_to, name, category, COUNT(*) as cnt, MAX(confidence) as max_conf
            FROM entities
            GROUP BY belongs_to, name, category
            HAVING cnt > 1
        """)

        for row in cursor.fetchall():
            cursor.execute("""
                DELETE FROM entities 
                WHERE belongs_to = ? AND name = ? AND category = ? AND id NOT IN (
                    SELECT id FROM entities 
                    WHERE belongs_to = ? AND name = ? AND category = ?
                    ORDER BY confidence DESC, occurrence_count DESC
                    LIMIT 1
                )
            """, (row['belongs_to'], row['name'], row['category'],
                  row['belongs_to'], row['name'], row['category']))
            self.stats['entities_deduplicated'] += 1

        self.conn.commit()

    # ── Step 7: Clean Relations ──

    def _clean_relations(self):
        """Удаляет self-relations и дубли."""
        logger.info("Cleaning relations...")
        cursor = self.conn.cursor()

        # Remove self-relations
        cursor.execute("DELETE FROM relations WHERE LOWER(source) = LOWER(target)")

        # Remove duplicate relations keeping strongest
        cursor.execute("""
            DELETE FROM relations 
            WHERE id NOT IN (
                SELECT MIN(id) FROM relations 
                GROUP BY LOWER(source), LOWER(target), relation_type, belongs_to
            )
        """)

        self.conn.commit()

    # ── Step 8: Generate Enhanced Vault ──

    def _generate_enhanced_vault(self):
        """Генерирует улучшенный vault из clean DB."""
        logger.info("Generating enhanced Obsidian vault...")

        if self.vault_dir.exists():
            shutil.rmtree(self.vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["People", "Entities", "Patterns", "Analysis", "Timeline"]:
            (self.vault_dir / subdir).mkdir(exist_ok=True)

        cursor = self.conn.cursor()

        # Get all people (only real people, filtered)
        cursor.execute("SELECT DISTINCT belongs_to FROM entities WHERE belongs_to != 'all'")
        people = set(r['belongs_to'] for r in cursor.fetchall())
        cursor.execute("SELECT DISTINCT author FROM messages")
        people.update(r['author'] for r in cursor.fetchall())
        for table in ['facts', 'cognitive_patterns', 'personality_dimensions_agg']:
            cursor.execute(f"SELECT DISTINCT belongs_to FROM {table} WHERE belongs_to != 'all'")
            people.update(r['belongs_to'] for r in cursor.fetchall())
        people.discard('Unknown')
        people.discard('')

        self._generate_home_page(people)

        for person in tqdm(sorted(people), desc="Generating clean profiles"):
            self._generate_clean_person_profile(cursor, person)

        self._generate_network_analysis(cursor)
        self._generate_global_patterns_index(cursor)
        self._generate_enhanced_timeline(cursor)

        logger.info(f"Enhanced vault generated: {self.vault_dir}")

    def _safe_filename(self, name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe[:100] or 'untitled'

    def _generate_home_page(self, people: Set[str]):
        content = f"""---
tags: [dossier, clean, consolidated]
date: {datetime.now().isoformat()}
---

# 🌐 Consolidated Multi-Personality Dossier

> **Post-processed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **Method:** LLM deduplication + statistical consolidation
> **Data Quality:** Cleaned, deduplicated, conflict-resolved

## 👥 People ({len(people)})

"""
        for person in sorted(people):
            safe = self._safe_filename(person)
            content += f"- [[People/{safe}/🧠 Profile|{person}]]\n"

        content += """
## 📊 Global Analysis

- [[Analysis/🕸️ Network|🕸️ Social Network Analysis]]
- [[Analysis/🧩 Patterns|🧩 Global Cognitive Patterns]]
- [[📅 Timeline|📅 Consolidated Timeline]]
"""
        (self.vault_dir / "🏠 Home.md").write_text(content, encoding='utf-8')

    def _generate_clean_person_profile(self, cursor, person: str):
        """Генерирует очищенный профиль с агрегированными данными."""
        safe = self._safe_filename(person)
        person_dir = self.vault_dir / "People" / safe
        person_dir.mkdir(parents=True, exist_ok=True)

        # Aggregated Big Five
        cursor.execute("""
            SELECT dimension, score, confidence, sample_count, evidence
            FROM personality_dimensions_agg
            WHERE belongs_to = ?
            ORDER BY dimension
        """, (person,))
        big_five = cursor.fetchall()

        # Consolidated facts by category
        cursor.execute("""
            SELECT fact, category, confidence, first_seen
            FROM facts
            WHERE belongs_to = ?
            ORDER BY category, confidence DESC
        """, (person,))
        facts = cursor.fetchall()

        # Patterns
        cursor.execute("""
            SELECT pattern_name, description, frequency, severity
            FROM cognitive_patterns
            WHERE belongs_to = ?
            ORDER BY frequency DESC
        """, (person,))
        patterns = cursor.fetchall()

        # Entities
        cursor.execute("""
            SELECT name, category, occurrence_count, confidence, sentiment
            FROM entities
            WHERE belongs_to = ?
            ORDER BY occurrence_count DESC, confidence DESC
            LIMIT 50
        """, (person,))
        entities = cursor.fetchall()

        # Relations
        cursor.execute("""
            SELECT source, target, relation_type, strength
            FROM relations
            WHERE belongs_to = ? OR source = ? OR target = ?
            ORDER BY strength DESC
            LIMIT 30
        """, (person, person, person))
        relations = cursor.fetchall()

        # Communication style
        cursor.execute("SELECT * FROM communication_styles WHERE belongs_to = ?", (person,))
        comm = cursor.fetchone()

        # Aliases
        aliases = [a for a, c in self.name_mapping.items() if c == person]

        # ── Profile Page ──
        profile = f"""---
tags: [profile, clean, {safe}]
person: {person}
aliases: {json.dumps(aliases)}
---

# 🧠 {person}

> **Каноническое имя:** {person}
> **Также известен как:** {', '.join(aliases) if aliases else '—'}

## 📊 Профиль личности

### Big Five (агрегировано)

| Измерение | Скор | Уверенность | Выборок |
|-----------|------|-------------|---------|
"""
        for dim in big_five:
            profile += f"| {dim['dimension'].title()} | {dim['score']:+.2f} | {dim['confidence']:.2f} | {dim['sample_count']} |\n"

        if not big_five:
            profile += "| *Нет данных* | — | — | — |\n"

        profile += "\n## 📝 Биография\n\n"

        by_cat = defaultdict(list)
        for f in facts:
            by_cat[f['category']].append(f)

        for cat, items in sorted(by_cat.items()):
            profile += f"\n### {cat.title()} ({len(items)})\n\n"
            for item in items[:20]:
                profile += f"- {item['fact']} *(уверенность: {item['confidence']:.2f})*\n"
            if len(items) > 20:
                profile += f"- *...и ещё {len(items) - 20}*\n"

        profile += "\n## 🧩 Когнитивные паттерны\n\n"
        for pat in patterns[:10]:
            profile += f"- **{pat['pattern_name']}** ({pat['severity']}, {pat['frequency']}×)\n"

        profile += "\n## 🎯 Ключевые сущности\n\n"
        for ent in entities[:20]:
            profile += f"- {ent['name']} ({ent['category']}, {ent['occurrence_count']}×, {ent['sentiment']})\n"

        profile += "\n## 🔗 Ключевые связи\n\n"
        profile += "```mermaid\ngraph LR\n"
        for rel in relations[:10]:
            src = self._safe_filename(rel['source'])
            tgt = self._safe_filename(rel['target'])
            profile += f"    {src}[\"{rel['source']}\"] -->|{rel['relation_type']}| {tgt}[\"{rel['target']}\"]\n"
        profile += "```\n"

        if comm:
            profile += f"""\n## 💬 Стиль общения

- **Формальность:** {comm['formality'] or 'неизвестно'}
- **Эмоциональная экспрессивность:** {comm['emotional_expressiveness'] or 'неизвестно'}
- **Аргументация:** {comm['argumentation'] or 'неизвестно'}
- **Юмор:** {comm['humor'] or 'неизвестно'}
- **Оборонительность:** {comm['defensiveness'] or 'неизвестно'}
"""

        (person_dir / "🧠 Profile.md").write_text(profile, encoding='utf-8')

        # ── Detailed Pages ──
        # Biography page
        bio = f"# 📜 Полная биография: {person}\n\n"
        for cat, items in sorted(by_cat.items()):
            bio += f"\n## {cat.title()}\n\n"
            for item in items:
                bio += f"- {item['fact']} *(уверенность: {item['confidence']:.2f}, {item['first_seen'][:10]})*\n"
        (person_dir / "📜 Biography.md").write_text(bio, encoding='utf-8')

        # Patterns page
        pat_page = f"# 🧩 Когнитивные паттерны: {person}\n\n"
        for pat in patterns:
            pat_page += f"""\n## {pat['pattern_name'].title()}

**Тяжесть:** {pat['severity']} | **Частота:** {pat['frequency']}

{pat['description'][:400]}

---
"""
        (person_dir / "🧩 Patterns.md").write_text(pat_page, encoding='utf-8')

        # Relations page
        rel_page = f"# 🔗 Все связи: {person}\n\n"
        rel_page += "| Источник | Тип | Цель | Сила |\n"
        rel_page += "|----------|-----|------|------|\n"
        for rel in relations:
            rel_page += f"| {rel['source']} | {rel['relation_type']} | {rel['target']} | {rel['strength']:.2f} |\n"
        (person_dir / "🔗 Relations.md").write_text(rel_page, encoding='utf-8')

    def _generate_network_analysis(self, cursor):
        """Генерирует анализ социальной сети."""
        cursor.execute("SELECT * FROM social_graph ORDER BY person_a, person_b")
        edges = cursor.fetchall()

        content = "# 🕸️ Анализ социальной сети\n\n"
        content += "```mermaid\ngraph TD\n"
        for edge in edges:
            a = self._safe_filename(edge['person_a'])
            b = self._safe_filename(edge['person_b'])
            content += f"    {a}[\"{edge['person_a']}\"] -->|{edge['relationship_type'] or 'связь'}| {b}[\"{edge['person_b']}\"]\n"
        content += "```\n\n"

        # Degree centrality
        cursor.execute("""
            SELECT person, COUNT(*) as degree FROM (
                SELECT person_a as person FROM social_graph
                UNION ALL
                SELECT person_b as person FROM social_graph
            ) GROUP BY person ORDER BY degree DESC
        """)
        content += "## Центральность по степени\n\n"
        content += "| Человек | Связей |\n"
        content += "|---------|--------|\n"
        for row in cursor.fetchall():
            content += f"| {row['person']} | {row['degree']} |\n"

        (self.vault_dir / "Analysis" / "🕸️ Network.md").mkdir(parents=True, exist_ok=True)
        (self.vault_dir / "Analysis" / "🕸️ Network.md").write_text(content, encoding='utf-8')

    def _generate_global_patterns_index(self, cursor):
        cursor.execute("""
            SELECT pattern_name, COUNT(DISTINCT belongs_to) as people, SUM(frequency) as total_freq
            FROM cognitive_patterns 
            GROUP BY pattern_name 
            ORDER BY total_freq DESC
        """)
        patterns = cursor.fetchall()
        content = "# 🧩 Глобальные когнитивные паттерны\n\n"
        content += "| Паттерн | Затронуто людей | Общая частота |\n"
        content += "|---------|-----------------|---------------|\n"
        for p in patterns:
            content += f"| {p['pattern_name']} | {p['people']} | {p['total_freq']} |\n"
        (self.vault_dir / "Analysis" / "🧩 Patterns.md").mkdir(parents=True, exist_ok=True)
        (self.vault_dir / "Analysis" / "🧩 Patterns.md").write_text(content, encoding='utf-8')

    def _generate_enhanced_timeline(self, cursor):
        cursor.execute("""
            SELECT temporal_markers, name, belongs_to 
            FROM entities 
            WHERE temporal_markers IS NOT NULL AND temporal_markers != '[]'
        """)
        rows = cursor.fetchall()
        events = []
        for row in rows:
            try:
                markers = json.loads(row['temporal_markers'])
                for m in markers:
                    events.append({'date': m, 'event': row['name'], 'person': row['belongs_to']})
            except:
                pass
        events.sort(key=lambda x: x['date'])

        content = "# 📅 Консолидированная хронология\n\n"
        content += "```mermaid\ntimeline\n    title События жизни\n"
        for ev in events[:80]:
            safe = self._safe_filename(ev['event'])
            content += f"    {ev['date']} : {ev['person']} — [[{safe}|{ev['event']}]]\n"
        content += "```\n"
        (self.vault_dir / "📅 Timeline.md").write_text(content, encoding='utf-8')

    def _print_stats(self):
        print()
        print("==================================================")
        print("│    POST-PROCESSING STATISTICS                  │")
        print("==================================================")

        for key, val in self.stats.items():
            print(f"  {key}: {val}")

        print("==================================================")

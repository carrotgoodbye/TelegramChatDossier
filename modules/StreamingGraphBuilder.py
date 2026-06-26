import json
import re
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from tqdm import tqdm


def ensure_list(val: Any) -> List[Any]:
    """Гарантирует, что значение — список."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # Иногда LLM возвращает строку вместо списка
        return []
    return [val]


def ensure_dict(val: Any) -> Dict[str, Any]:
    """Гарантирует, что значение — dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    return {}


class StreamingGraphBuilder:
    def __init__(self, output_dir: str, all_authors: Set[str]):
        self.output_dir = Path(output_dir)
        self.all_authors = all_authors
        self.vault_dir = self.output_dir / "obsidian_vault"
        self.db_path = self.output_dir / "personality_graph.db"

        self.vault_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["People", "Entities", "Patterns", "Events", "Relations", "Timeline"]:
            (self.vault_dir / subdir).mkdir(exist_ok=True)

        self._init_db()

        self._entity_keys: Set[str] = set()
        self._relation_keys: Set[str] = set()
        self._pattern_keys: Set[str] = set()
        self._fact_hashes: Set[str] = set()
        self._dimension_keys: Set[str] = set()

        self._load_existing_keys()

        self.stats = {
            'chunks_processed': 0,
            'entities_added': 0, 'entities_merged': 0,
            'relations_added': 0, 'relations_merged': 0,
            'patterns_added': 0, 'facts_added': 0, 'dimensions_added': 0,
        }

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                context TEXT,
                confidence REAL,
                sentiment TEXT,
                temporal_markers TEXT,
                cognitive_pattern TEXT,
                belongs_to TEXT,
                first_seen TEXT,
                last_seen TEXT,
                occurrence_count INTEGER DEFAULT 1,
                source_message_ids TEXT,
                UNIQUE(name, belongs_to)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_belongs ON entities(belongs_to);
            CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                relation_type TEXT,
                evidence TEXT,
                strength REAL,
                belongs_to TEXT,
                source_message_ids TEXT,
                UNIQUE(source, target, relation_type, belongs_to)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_belongs ON relations(belongs_to);

            CREATE TABLE IF NOT EXISTS cognitive_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                description TEXT,
                frequency INTEGER DEFAULT 1,
                evidence_messages TEXT,
                triggers TEXT,
                associated_emotions TEXT,
                severity TEXT,
                belongs_to TEXT,
                UNIQUE(pattern_name, belongs_to)
            );
            CREATE INDEX IF NOT EXISTS idx_patterns_belongs ON cognitive_patterns(belongs_to);

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT,
                category TEXT,
                confidence REAL,
                belongs_to TEXT,
                first_seen TEXT,
                source_message_ids TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_facts_belongs ON facts(belongs_to);
            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);

            CREATE TABLE IF NOT EXISTS personality_dimensions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dimension TEXT,
                score REAL,
                evidence TEXT,
                confidence REAL,
                belongs_to TEXT,
                timestamp TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dims_belongs ON personality_dimensions(belongs_to);

            CREATE TABLE IF NOT EXISTS communication_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                belongs_to TEXT UNIQUE,
                formality TEXT,
                emotional_expressiveness TEXT,
                argumentation TEXT,
                humor TEXT,
                defensiveness TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS social_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_a TEXT,
                person_b TEXT,
                relationship_type TEXT,
                emotional_tone TEXT,
                power_dynamic TEXT,
                frequency TEXT,
                evidence TEXT,
                UNIQUE(person_a, person_b)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                date TEXT,
                author TEXT,
                text TEXT,
                sentiment TEXT
            );

            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                timestamp TEXT,
                messages_processed INTEGER,
                status TEXT
            );
        """)
        self.conn.commit()

    def _load_existing_keys(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, belongs_to FROM entities")
        for row in cursor.fetchall():
            self._entity_keys.add(f"{row['name']}::{row['belongs_to']}")
        cursor.execute("SELECT source, target, relation_type, belongs_to FROM relations")
        for row in cursor.fetchall():
            self._relation_keys.add(f"{row['source']}::{row['target']}::{row['relation_type']}::{row['belongs_to']}")
        cursor.execute("SELECT pattern_name, belongs_to FROM cognitive_patterns")
        for row in cursor.fetchall():
            self._pattern_keys.add(f"{row['pattern_name']}::{row['belongs_to']}")
        cursor.execute("SELECT fact, belongs_to FROM facts")
        for row in cursor.fetchall():
            self._fact_hashes.add(hashlib.md5(f"{row['fact']}::{row['belongs_to']}".encode()).hexdigest())
        cursor.execute("SELECT dimension, belongs_to FROM personality_dimensions")
        for row in cursor.fetchall():
            self._dimension_keys.add(f"{row['dimension']}::{row['belongs_to']}")
        print(f"[Builder] Loaded existing: {len(self._entity_keys)} entities, {len(self._relation_keys)} relations, "
              f"{len(self._pattern_keys)} patterns from previous runs")

    def process_chunk(self, chunk_id: int, chunk_meta: Dict, result: Dict[str, Any]):
        """Обрабатывает один чанк и сразу коммитит в БД."""
        # Защита: result должен быть dict
        if not isinstance(result, dict):
            print(f"[WARN] Chunk {chunk_id}: result is {type(result).__name__}, not dict. Skipping.")
            return

        msg_ids = chunk_meta.get('message_ids', [])
        timestamp = chunk_meta.get('date_to', datetime.now().isoformat())
        cursor = self.conn.cursor()

        # Используем ensure_list для всех списков — защита от строк вместо списков от LLM
        for ent in ensure_list(result.get('entities')):
            if isinstance(ent, dict):
                self._upsert_entity(cursor, ent, msg_ids, timestamp)

        for rel in ensure_list(result.get('relations')):
            if isinstance(rel, dict):
                self._upsert_relation(cursor, rel, msg_ids)

        for pat in ensure_list(result.get('cognitive_patterns')):
            if isinstance(pat, dict):
                self._upsert_pattern(cursor, pat, msg_ids)

        for fact in ensure_list(result.get('facts')):
            if isinstance(fact, dict):
                self._insert_fact(cursor, fact, msg_ids, timestamp)

        for dim in ensure_list(result.get('personality_dimensions')):
            if isinstance(dim, dict):
                self._insert_dimension(cursor, dim, timestamp)

        for style in ensure_list(result.get('communication_styles')):
            if isinstance(style, dict):
                self._upsert_comm_style(cursor, style, timestamp)

        for edge in ensure_list(result.get('social_graph')):
            if isinstance(edge, dict):
                self._upsert_social_edge(cursor, edge)

        self._store_messages(cursor, chunk_meta.get('raw_messages', []))

        cursor.execute("""
            INSERT INTO processing_log (chunk_id, timestamp, messages_processed, status)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, timestamp, len(msg_ids), 'success'))

        self.conn.commit()
        self.stats['chunks_processed'] += 1

    def _upsert_entity(self, cursor, ent: Dict, msg_ids: List[int], timestamp: str):
        name = ent.get('name', '').strip()
        if not name or len(name) < 2:
            return
        belongs_to = ent.get('belongs_to', 'all')
        if not isinstance(belongs_to, str):
            belongs_to = 'all'
        category = ent.get('category', 'unknown')
        if not isinstance(category, str):
            category = 'unknown'
        category = category.lower()
        key = f"{name.lower()}::{belongs_to}"

        if key in self._entity_keys:
            cursor.execute("""
                UPDATE entities SET
                    occurrence_count = occurrence_count + 1,
                    confidence = MAX(confidence, ?),
                    last_seen = MAX(last_seen, ?),
                    context = COALESCE(context, '') || ' | ' || ?,
                    source_message_ids = COALESCE(source_message_ids, '') || ',' || ?
                WHERE name = ? AND belongs_to = ?
            """, (float(ent.get('confidence', 0.5)), timestamp, str(ent.get('context', ''))[:300],
                  json.dumps(msg_ids), name, belongs_to))
            self.stats['entities_merged'] += 1
        else:
            cursor.execute("""
                INSERT INTO entities (name, category, context, confidence, sentiment,
                    temporal_markers, cognitive_pattern, belongs_to, first_seen, last_seen,
                    occurrence_count, source_message_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, category, str(ent.get('context', ''))[:500], float(ent.get('confidence', 0.5)),
                  str(ent.get('sentiment', 'neutral')), json.dumps(ensure_list(ent.get('temporal_markers')))[:10],
                  str(ent.get('cognitive_pattern', '')), belongs_to, timestamp, timestamp, 1, json.dumps(msg_ids)))
            self._entity_keys.add(key)
            self.stats['entities_added'] += 1

    def _upsert_relation(self, cursor, rel: Dict, msg_ids: List[int]):
        source = str(rel.get('source', '')).strip()
        target = str(rel.get('target', '')).strip()
        rel_type = str(rel.get('relation_type', 'related_to')).lower()
        belongs_to = str(rel.get('belongs_to', 'all'))
        if not source or not target or source == target:
            return
        key = f"{source.lower()}::{target.lower()}::{rel_type}::{belongs_to}"

        if key in self._relation_keys:
            cursor.execute("""
                UPDATE relations SET
                    strength = MAX(strength, ?),
                    evidence = COALESCE(evidence, '') || ' | ' || ?,
                    source_message_ids = COALESCE(source_message_ids, '') || ',' || ?
                WHERE source = ? AND target = ? AND relation_type = ? AND belongs_to = ?
            """, (float(rel.get('strength', 0.5)), str(rel.get('evidence', ''))[:300],
                  json.dumps(msg_ids), source, target, rel_type, belongs_to))
            self.stats['relations_merged'] += 1
        else:
            cursor.execute("""
                INSERT INTO relations (source, target, relation_type, evidence, strength, belongs_to, source_message_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source, target, rel_type, str(rel.get('evidence', ''))[:500],
                  float(rel.get('strength', 0.5)), belongs_to, json.dumps(msg_ids)))
            self._relation_keys.add(key)
            self.stats['relations_added'] += 1

    def _upsert_pattern(self, cursor, pat: Dict, msg_ids: List[int]):
        name = str(pat.get('pattern_name', '')).strip().lower()
        if not name:
            return
        belongs_to = str(pat.get('belongs_to', 'all'))
        key = f"{name}::{belongs_to}"

        if key in self._pattern_keys:
            cursor.execute("""
                UPDATE cognitive_patterns SET
                    frequency = frequency + 1,
                    description = COALESCE(description, '') || ' | ' || ?,
                    evidence_messages = COALESCE(evidence_messages, '') || ',' || ?
                WHERE pattern_name = ? AND belongs_to = ?
            """, (str(pat.get('description', ''))[:300], json.dumps(msg_ids), pat.get('pattern_name', name), belongs_to))
        else:
            cursor.execute("""
                INSERT INTO cognitive_patterns (pattern_name, description, frequency, evidence_messages,
                    triggers, associated_emotions, severity, belongs_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pat.get('pattern_name', name), str(pat.get('description', ''))[:500], 1,
                  json.dumps(msg_ids), json.dumps(ensure_list(pat.get('triggers'))),
                  json.dumps(ensure_list(pat.get('associated_emotions'))), str(pat.get('severity', 'mild')), belongs_to))
            self._pattern_keys.add(key)
            self.stats['patterns_added'] += 1

    def _insert_fact(self, cursor, fact: Dict, msg_ids: List[int], timestamp: str):
        text = str(fact.get('fact', '')).strip()
        if not text:
            return
        belongs_to = str(fact.get('belongs_to', 'all'))
        fact_hash = hashlib.md5(f"{text}::{belongs_to}".encode()).hexdigest()
        if fact_hash in self._fact_hashes:
            return
        cursor.execute("""
            INSERT INTO facts (fact, category, confidence, belongs_to, first_seen, source_message_ids)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (text[:1000], str(fact.get('category', 'general')), float(fact.get('confidence', 0.5)),
              belongs_to, timestamp, json.dumps(msg_ids)))
        self._fact_hashes.add(fact_hash)
        self.stats['facts_added'] += 1

    def _insert_dimension(self, cursor, dim: Dict, timestamp: str):
        dimension = str(dim.get('dimension', '')).strip().lower()
        if not dimension:
            return
        belongs_to = str(dim.get('belongs_to', 'all'))
        cursor.execute("""
            INSERT INTO personality_dimensions (dimension, score, evidence, confidence, belongs_to, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (dimension, float(dim.get('score', 0.0)), str(dim.get('evidence', ''))[:500],
              float(dim.get('confidence', 0.5)), belongs_to, timestamp))
        self.stats['dimensions_added'] += 1

    def _upsert_comm_style(self, cursor, style: Dict, timestamp: str):
        belongs_to = str(style.get('belongs_to', 'all'))
        cursor.execute("""
            INSERT INTO communication_styles (belongs_to, formality, emotional_expressiveness,
                argumentation, humor, defensiveness, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(belongs_to) DO UPDATE SET
                formality = COALESCE(excluded.formality, formality),
                emotional_expressiveness = COALESCE(excluded.emotional_expressiveness, emotional_expressiveness),
                argumentation = COALESCE(excluded.argumentation, argumentation),
                humor = COALESCE(excluded.humor, humor),
                defensiveness = COALESCE(excluded.defensiveness, defensiveness),
                updated_at = excluded.updated_at
        """, (belongs_to, str(style.get('formality', '')), str(style.get('emotional_expressiveness', '')),
              str(style.get('argumentation', '')), str(style.get('humor', '')), str(style.get('defensiveness', '')), timestamp))

    def _upsert_social_edge(self, cursor, edge: Dict):
        a = str(edge.get('person_a', '')).strip()
        b = str(edge.get('person_b', '')).strip()
        if not a or not b:
            return
        if a > b:
            a, b = b, a
        cursor.execute("""
            INSERT INTO social_graph (person_a, person_b, relationship_type, emotional_tone, power_dynamic, frequency, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_a, person_b) DO UPDATE SET
                relationship_type = COALESCE(excluded.relationship_type, relationship_type),
                emotional_tone = COALESCE(excluded.emotional_tone, emotional_tone),
                power_dynamic = COALESCE(excluded.power_dynamic, power_dynamic),
                frequency = COALESCE(excluded.frequency, frequency),
                evidence = COALESCE(evidence, '') || ' | ' || excluded.evidence
        """, (a, b, str(edge.get('relationship_type', '')), str(edge.get('emotional_tone', '')),
              str(edge.get('power_dynamic', '')), str(edge.get('frequency', '')), str(edge.get('evidence', ''))[:300]))

    def _store_messages(self, cursor, messages: List[Dict]):
        for msg in messages:
            msg_id = msg.get('id')
            if not msg_id:
                continue
            text = msg.get('text', '')
            if isinstance(text, list):
                text = ' '.join(str(p) if isinstance(p, str) else p.get('text', '') for p in text)
            cursor.execute("""
                INSERT OR IGNORE INTO messages (id, date, author, text)
                VALUES (?, ?, ?, ?)
            """, (msg_id, msg.get('date'), msg.get('from') or msg.get('sender_name'), text))

    def print_stats(self):
        print("\n" + "=" * 50)
        print("PROCESSING STATISTICS")
        print("=" * 50)
        for key, val in self.stats.items():
            print(f"  {key}: {val}")
        print("=" * 50)

    # ── Obsidian Vault Generation ──

    def generate_obsidian_vault(self):
        print("[Vault] Generating Obsidian vault from database...")
        cursor = self.conn.cursor()

        cursor.execute("SELECT DISTINCT belongs_to FROM entities WHERE belongs_to != 'all'")
        people = set(row['belongs_to'] for row in cursor.fetchall())
        cursor.execute("SELECT DISTINCT author FROM messages")
        people.update(row['author'] for row in cursor.fetchall())
        for table in ['facts', 'cognitive_patterns', 'personality_dimensions']:
            cursor.execute(f"SELECT DISTINCT belongs_to FROM {table} WHERE belongs_to != 'all'")
            people.update(row['belongs_to'] for row in cursor.fetchall())
        people.discard('Unknown')
        people.discard('')

        print(f"[Vault] Found {len(people)} people to profile")
        self._generate_main_page(people)
        for person in tqdm(sorted(people), desc="Generating profiles"):
            self._generate_person_tree(cursor, person)
        self._generate_entities_index(cursor)
        self._generate_patterns_index(cursor)
        self._generate_social_graph_page(cursor)
        self._generate_timeline(cursor)
        print(f"[Vault] Done: {self.vault_dir}")

    def _safe_filename(self, name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe[:100] or 'untitled'

    def _generate_main_page(self, people: Set[str]):
        content = f"""---
tags: [dossier, multi-personality-graph]
date: {datetime.now().isoformat()}
---

# 🌐 Multi-Personality Dossier

> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **Source:** Telegram Chat Export
> **Method:** LLM-based multi-person extraction

## 👥 People in this Network ({len(people)})

"""
        for person in sorted(people):
            safe = self._safe_filename(person)
            content += f"- [[{safe}/🧠 Profile|{person}]]\n"

        content += """
## 🗺️ Network Overview

```mermaid
mindmap
  root((Social Network))
"""
        for person in sorted(people)[:15]:
            safe = self._safe_filename(person)
            content += f"    {safe}[{person}]\n"
        content += "```\n"

        content += """
## 📂 Indexes

- [[All Entities|📦 All Entities]]
- [[Cognitive Patterns Index|🧩 Cognitive Patterns]]
- [[Social Graph|🕸️ Social Graph]]
- [[Timeline|📅 Timeline]]
"""
        (self.vault_dir / "🏠 Home.md").write_text(content, encoding='utf-8')

    def _generate_person_tree(self, cursor, person: str):
        safe = self._safe_filename(person)
        person_dir = self.vault_dir / "People" / safe
        person_dir.mkdir(parents=True, exist_ok=True)

        (person_dir / "🧠 Profile.md").write_text(self._build_profile_page(cursor, person), encoding='utf-8')
        (person_dir / "📜 Biography.md").write_text(self._build_bio_page(cursor, person), encoding='utf-8')
        (person_dir / "🧩 Patterns.md").write_text(self._build_patterns_page(cursor, person), encoding='utf-8')
        (person_dir / "📊 Dimensions.md").write_text(self._build_dimensions_page(cursor, person), encoding='utf-8')
        (person_dir / "🎯 Entities.md").write_text(self._build_entities_page(cursor, person), encoding='utf-8')
        (person_dir / "🔗 Relations.md").write_text(self._build_relations_page(cursor, person), encoding='utf-8')
        (person_dir / "💬 Communication.md").write_text(self._build_comm_page(cursor, person), encoding='utf-8')

    def _build_profile_page(self, cursor, person: str) -> str:
        cursor.execute("SELECT COUNT(*) as cnt FROM entities WHERE belongs_to = ?", (person,))
        entity_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) as cnt FROM facts WHERE belongs_to = ?", (person,))
        fact_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) as cnt FROM cognitive_patterns WHERE belongs_to = ?", (person,))
        pattern_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) as cnt FROM relations WHERE belongs_to = ? OR source = ? OR target = ?",
                      (person, person, person))
        relation_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT * FROM communication_styles WHERE belongs_to = ?", (person,))
        comm = cursor.fetchone()

        content = f"""---
tags: [profile, person, {self._safe_filename(person)}]
person: {person}
---

# 🧠 {person}

> Central profile node for **{person}**

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Entities | {entity_count} |
| Facts | {fact_count} |
| Cognitive Patterns | {pattern_count} |
| Relations | {relation_count} |

## 🌳 Profile Tree

- [[📜 Biography|📜 Biography & Facts]]
- [[🧩 Patterns|🧩 Cognitive Patterns]]
- [[📊 Dimensions|📊 Personality Dimensions]]
- [[🎯 Entities|🎯 Entities & Interests]]
- [[🔗 Relations|🔗 Relations & Connections]]
- [[💬 Communication|💬 Communication Style]]

## 💬 Communication Style

"""
        if comm:
            content += f"""- **Formality:** {comm['formality'] or 'unknown'}
- **Emotional Expressiveness:** {comm['emotional_expressiveness'] or 'unknown'}
- **Argumentation:** {comm['argumentation'] or 'unknown'}
- **Humor:** {comm['humor'] or 'unknown'}
- **Defensiveness:** {comm['defensiveness'] or 'unknown'}
"""
        else:
            content += "*No communication style data extracted yet.*\n"

        content += f"""
## 🗺️ Personal Mind Map

```mermaid
mindmap
  root(({person}))
    Biography
      Facts
      Events
    Psychology
      Patterns
      Dimensions
    Social
      Relations
      Network
    Interests
      Entities
      Preferences
```
"""
        return content

    def _build_bio_page(self, cursor, person: str) -> str:
        cursor.execute("""
            SELECT fact, category, confidence, first_seen 
            FROM facts 
            WHERE belongs_to = ? 
            ORDER BY confidence DESC, first_seen DESC
        """, (person,))
        facts = cursor.fetchall()
        content = f"# 📜 Biography: {person}\n\n"
        by_cat = defaultdict(list)
        for f in facts:
            by_cat[f['category']].append(f)
        for cat, items in sorted(by_cat.items()):
            content += f"\n## {cat.title()}\n\n"
            for item in items:
                content += f"- {item['fact']} *(confidence: {item['confidence']:.2f}, {item['first_seen'][:10]})*\n"
        if not facts:
            content += "\n*No biographical facts extracted yet.*\n"
        return content

    def _build_patterns_page(self, cursor, person: str) -> str:
        cursor.execute("""
            SELECT pattern_name, description, frequency, severity, triggers, associated_emotions
            FROM cognitive_patterns 
            WHERE belongs_to = ? 
            ORDER BY frequency DESC
        """, (person,))
        patterns = cursor.fetchall()
        content = f"# 🧩 Cognitive Patterns: {person}\n\n"
        for pat in patterns:
            content += f"""\n## {pat['pattern_name'].title()}

**Severity:** {pat['severity']} | **Frequency:** {pat['frequency']} mentions

**Description:** {pat['description'][:300]}

**Triggers:** {pat['triggers']}

**Associated Emotions:** {pat['associated_emotions']}

---
"""
        if not patterns:
            content += "\n*No cognitive patterns identified yet.*\n"
        return content

    def _build_dimensions_page(self, cursor, person: str) -> str:
        cursor.execute("""
            SELECT dimension, score, evidence, confidence, timestamp
            FROM personality_dimensions 
            WHERE belongs_to = ? 
            ORDER BY timestamp DESC
        """, (person,))
        dims = cursor.fetchall()
        content = f"# 📊 Personality Dimensions: {person}\n\n"
        content += "| Dimension | Score | Confidence | Evidence | Date |\n"
        content += "|-----------|-------|------------|----------|------|\n"
        for d in dims:
            content += f"| {d['dimension'].title()} | {d['score']:+.2f} | {d['confidence']:.2f} | {d['evidence'][:40]}... | {d['timestamp'][:10]} |\n"
        if not dims:
            content += "\n*No personality dimensions inferred yet.*\n"
        return content

    def _build_entities_page(self, cursor, person: str) -> str:
        cursor.execute("""
            SELECT name, category, context, confidence, sentiment, occurrence_count
            FROM entities 
            WHERE belongs_to = ? 
            ORDER BY occurrence_count DESC, confidence DESC
        """, (person,))
        entities = cursor.fetchall()
        content = f"# 🎯 Entities & Interests: {person}\n\n"
        by_cat = defaultdict(list)
        for e in entities:
            by_cat[e['category']].append(e)
        for cat, items in sorted(by_cat.items()):
            content += f"\n## {cat.title()}\n\n"
            for e in items[:30]:
                safe = self._safe_filename(e['name'])
                content += f"- [[{safe}|{e['name']}]] — *{e['sentiment']}* (conf: {e['confidence']:.2f}, {e['occurrence_count']}×)\n"
            if len(items) > 30:
                content += f"- *...and {len(items) - 30} more*\n"
        if not entities:
            content += "\n*No entities extracted yet.*\n"
        return content

    def _build_relations_page(self, cursor, person: str) -> str:
        cursor.execute("""
            SELECT source, target, relation_type, evidence, strength
            FROM relations 
            WHERE belongs_to = ? OR source = ? OR target = ?
            ORDER BY strength DESC
        """, (person, person, person))
        relations = cursor.fetchall()
        content = f"# 🔗 Relations & Connections: {person}\n\n"
        content += "```mermaid\ngraph LR\n"
        for rel in relations[:15]:
            src = self._safe_filename(rel['source'])
            tgt = self._safe_filename(rel['target'])
            content += f"    {src}[\"{rel['source']}\"] -->|{rel['relation_type']}| {tgt}[\"{rel['target']}\"]\n"
        content += "```\n\n"
        content += "## All Relations\n\n"
        content += "| Source | Relation | Target | Strength |\n"
        content += "|--------|----------|--------|----------|\n"
        for rel in relations:
            content += f"| {rel['source']} | {rel['relation_type']} | {rel['target']} | {rel['strength']:.2f} |\n"
        if not relations:
            content += "\n*No relations extracted yet.*\n"
        return content

    def _build_comm_page(self, cursor, person: str) -> str:
        cursor.execute("SELECT * FROM communication_styles WHERE belongs_to = ?", (person,))
        comm = cursor.fetchone()
        content = f"# 💬 Communication Style: {person}\n\n"
        if comm:
            content += f"""- **Formality:** {comm['formality']}
- **Emotional Expressiveness:** {comm['emotional_expressiveness']}
- **Argumentation:** {comm['argumentation']}
- **Humor:** {comm['humor']}
- **Defensiveness:** {comm['defensiveness']}
- **Last Updated:** {comm['updated_at']}
"""
        else:
            content += "*No data.*\n"
        cursor.execute("""
            SELECT date, text FROM messages 
            WHERE author = ? 
            ORDER BY date DESC 
            LIMIT 20
        """, (person,))
        msgs = cursor.fetchall()
        content += "\n## Recent Messages\n\n"
        for msg in msgs:
            text = msg['text'][:200] if msg['text'] else ''
            content += f"> [{msg['date'][:16]}] {text}...\n\n"
        return content

    def _generate_entities_index(self, cursor):
        cursor.execute("SELECT DISTINCT category FROM entities ORDER BY category")
        cats = [r['category'] for r in cursor.fetchall()]
        content = "# 📦 All Entities\n\n"
        for cat in cats:
            cursor.execute("""
                SELECT name, category, COUNT(*) as cnt 
                FROM entities 
                WHERE category = ? 
                GROUP BY name 
                ORDER BY cnt DESC 
                LIMIT 50
            """, (cat,))
            items = cursor.fetchall()
            content += f"\n## {cat.title()}\n\n"
            for item in items:
                safe = self._safe_filename(item['name'])
                content += f"- [[{safe}|{item['name']}]] ({item['cnt']} refs)\n"
        (self.vault_dir / "All Entities.md").write_text(content, encoding='utf-8')

    def _generate_patterns_index(self, cursor):
        cursor.execute("""
            SELECT pattern_name, COUNT(DISTINCT belongs_to) as people, SUM(frequency) as total_freq
            FROM cognitive_patterns 
            GROUP BY pattern_name 
            ORDER BY total_freq DESC
        """)
        patterns = cursor.fetchall()
        content = "# 🧩 Cognitive Patterns Index\n\n"
        content += "| Pattern | Affected People | Total Frequency |\n"
        content += "|---------|-----------------|-----------------|\n"
        for p in patterns:
            content += f"| {p['pattern_name']} | {p['people']} | {p['total_freq']} |\n"
        (self.vault_dir / "Cognitive Patterns Index.md").write_text(content, encoding='utf-8')

    def _generate_social_graph_page(self, cursor):
        cursor.execute("SELECT * FROM social_graph ORDER BY person_a, person_b")
        edges = cursor.fetchall()
        content = "# 🕸️ Social Graph\n\n"
        content += "```mermaid\ngraph TD\n"
        for edge in edges:
            a = self._safe_filename(edge['person_a'])
            b = self._safe_filename(edge['person_b'])
            content += f"    {a}[\"{edge['person_a']}\"] -->|{edge['relationship_type'] or 'connected'}| {b}[\"{edge['person_b']}\"]\n"
        content += "```\n\n"
        content += "## Edges\n\n"
        content += "| Person A | Relation | Person B | Tone | Power | Freq |\n"
        content += "|----------|----------|----------|------|-------|------|\n"
        for edge in edges:
            content += f"| {edge['person_a']} | {edge['relationship_type']} | {edge['person_b']} | {edge['emotional_tone']} | {edge['power_dynamic']} | {edge['frequency']} |\n"
        (self.vault_dir / "Social Graph.md").write_text(content, encoding='utf-8')

    def _generate_timeline(self, cursor):
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
        content = "# 📅 Timeline\n\n"
        content += "```mermaid\ntimeline\n    title Events Timeline\n"
        for ev in events[:50]:
            safe = self._safe_filename(ev['event'])
            content += f"    {ev['date']} : {ev['person']} — [[{safe}|{ev['event']}]]\n"
        content += "```\n"
        (self.vault_dir / "Timeline.md").write_text(content, encoding='utf-8')

    def close(self):
        self.conn.close()

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set


def ensure_list(val: Any) -> List[Any]:
    """Гарантирует, что значение — список."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
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
        self.db_path = self.output_dir / "personality_graph.db"

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

    def close(self):
        self.conn.close()

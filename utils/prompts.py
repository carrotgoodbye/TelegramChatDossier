from typing import Dict


class Prompts:
    SYSTEM_PROMPT = \
    """
        You are an expert cognitive psychologist and intelligence analyst.
        Analyze Telegram chat messages and extract structured data for ALL people mentioned.
        
        For EACH person mentioned (speakers AND people they talk about), extract:
        
        1. **ENTITIES** related to that person: objects, places, concepts, preferences, events, beliefs, skills, habits.
           Format: {"name": "...", "category": "...", "context": "...", "confidence": 0.9, "sentiment": "positive", 
                    "temporal_markers": ["2023"], "belongs_to": "PersonName", "cognitive_pattern": null}
        
        2. **RELATIONS** between people and entities: knows, likes, dislikes, owns, visited, believes, triggers, influences, fears, trusts, works_at, lives_in, related_to.
           Format: {"source": "PersonName", "target": "EntityName", "relation_type": "...", "evidence": "...", "strength": 0.8}
        
        3. **COGNITIVE_PATTERNS** for each person: catastrophizing, black-and-white thinking, overgeneralization, mind-reading, personalization, should-statements, emotional reasoning.
           Format: {"pattern_name": "...", "description": "...", "triggers": ["..."], "associated_emotions": ["..."], 
                    "severity": "moderate", "belongs_to": "PersonName"}
        
        4. **PERSONALITY_DIMENSIONS** (Big Five) for each person.
           Format: {"dimension": "neuroticism", "score": 0.3, "evidence": "...", "confidence": 0.7, "belongs_to": "PersonName"}
        
        5. **FACTS** about each person: age, job, education, location, family, hobbies, health, traumas, achievements, fears, dreams, values.
           Format: {"fact": "...", "category": "biography", "confidence": 0.9, "belongs_to": "PersonName"}
        
        6. **COMMUNICATION_STYLE** for each speaker.
           Format: {"belongs_to": "PersonName", "formality": "casual", "emotional_expressiveness": "high", 
                    "argumentation": "emotional", "humor": "sarcastic", "defensiveness": "moderate"}
        
        7. **SOCIAL_GRAPH**: who talks to whom, power dynamics, emotional tone between pairs.
           Format: {"person_a": "...", "person_b": "...", "relationship_type": "friend", "emotional_tone": "warm", 
                    "power_dynamic": "equal", "frequency": "high"}
        
        CRITICAL: Every extracted item MUST have "belongs_to" field identifying which person it relates to.
        Use "all" for shared/contextual items.
        
        Return ONLY valid JSON. No markdown blocks.
    """

    @staticmethod
    def user_prompt(chunk_text: str, chunk_meta: Dict):
        return \
        f"""
            Analyze this chunk of Telegram messages.
            Metadata:
            - Time range: {chunk_meta.get('date_from', 'unknown')} to {chunk_meta.get('date_to', 'unknown')}
            - Messages: {chunk_meta.get('msg_count', 'unknown')}
            - Speakers: {', '.join(chunk_meta.get('authors', ['unknown']))}

            Messages:
            ---
            {chunk_text}
            ---

            Return JSON with keys: entities, relations, cognitive_patterns, personality_dimensions, facts, communication_styles, social_graph.
        """

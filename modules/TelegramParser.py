import json
import logging
from typing import List, Dict, Set

logger = logging.getLogger("TelegramParser")


class TelegramParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.messages: List[Dict] = []
        self.all_authors: Set[str] = set()

    def parse(self) -> List[Dict]:
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.messages = data.get('messages', [])
        elif isinstance(data, list):
            self.messages = data
        else:
            raise ValueError("Unsupported JSON format")

        self.messages = [
            msg for msg in self.messages
            if msg.get('type') == 'message' and self._extract_text(msg)
        ]
        self.messages.sort(key=lambda x: x.get('date', ''))

        for msg in self.messages:
            author = msg.get('from') or msg.get('sender_name') or 'Unknown'
            self.all_authors.add(author)

        logger.info(f"{len(self.messages)} messages, {len(self.all_authors)} unique authors: {self.all_authors}")
        return self.messages

    def _extract_text(self, msg: Dict) -> str:
        text = msg.get('text', '')
        if isinstance(text, list):
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get('text', ''))
            text = ' '.join(parts)
        return str(text).strip()

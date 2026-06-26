from typing import List, Dict


class Chunker:
    def __init__(self, max_chunk_size: int = 6000, overlap_messages: int = 2):
        self.max_chunk_size = max_chunk_size
        self.overlap_messages = overlap_messages

    def create_chunks(self, messages: List[Dict]) -> List[Dict]:
        chunks = []
        current_chunk = []
        current_size = 0

        for msg in messages:
            text = self._format_message(msg)
            msg_size = len(text)

            if current_size + msg_size > self.max_chunk_size and current_chunk:
                chunks.append(self._finalize_chunk(current_chunk))
                overlap = current_chunk[-self.overlap_messages:] if len(current_chunk) > self.overlap_messages else current_chunk
                current_chunk = overlap + [msg]
                current_size = sum(len(self._format_message(m)) for m in current_chunk)
            else:
                current_chunk.append(msg)
                current_size += msg_size

        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk))

        return chunks

    def _format_message(self, msg: Dict) -> str:
        author = msg.get('from') or msg.get('sender_name') or 'Unknown'
        date = msg.get('date', 'unknown')
        text = msg.get('text', '')
        if isinstance(text, list):
            text = ' '.join(str(p) if isinstance(p, str) else p.get('text', '') for p in text)
        reactions = msg.get('reactions', [])
        reaction_str = f" [reactions: {reactions}]" if reactions else ""
        return f"[{date}] {author}: {text}{reaction_str}\n"

    def _finalize_chunk(self, chunk_messages: List[Dict]) -> Dict:
        texts = [self._format_message(m) for m in chunk_messages]
        authors = list(set(m.get('from') or m.get('sender_name') or 'Unknown' for m in chunk_messages))
        dates = [m.get('date', '') for m in chunk_messages if m.get('date')]

        return {
            'text': ''.join(texts),
            'msg_count': len(chunk_messages),
            'authors': authors,
            'date_from': min(dates) if dates else 'unknown',
            'date_to': max(dates) if dates else 'unknown',
            'message_ids': [m.get('id', i) for i, m in enumerate(chunk_messages)],
            'raw_messages': chunk_messages
        }

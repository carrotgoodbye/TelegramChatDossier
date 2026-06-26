import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from modules.TelegramParser import TelegramParser
from modules.LLMProcessor import LLMProcessor
from modules.Chunker import Chunker
from modules.StreamingGraphBuilder import StreamingGraphBuilder

logger = logging.getLogger("Pipeline")


class Pipeline:
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        ollama_url: str,
        model: str,
        max_chunk_size: Optional[int] = None,
        target_user: Optional[str] = None
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.ollama_url = ollama_url
        self.model = model
        self.target_user = target_user

        self.parser = TelegramParser(input_file)
        self.llm = LLMProcessor(ollama_url=ollama_url, model=model)

        chunk_size = max_chunk_size or self.llm.max_chunk_chars
        self.chunker = Chunker(max_chunk_size=chunk_size, overlap_messages=2)
        logger.info(f"Chunk size: {chunk_size} chars")

    def run(self):
        print("┌────────────────────────────────────────┐")
        print("│                                        │")
        print("│    Multi-Personality Graph Builder     │")
        print("│                                        │")
        print("└────────────────────────────────────────┘")

        messages = self.parser.parse()
        if not messages:
            logger.error("No messages!")
            return

        builder = StreamingGraphBuilder(self.output_dir, self.parser.all_authors)
        chunks = self.chunker.create_chunks(messages)
        logger.info(f"{len(chunks)} chunks to process")

        # Проверяем прогресс
        cursor = builder.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM processing_log WHERE status = 'success'")
        completed = cursor.fetchone()['cnt']
        if completed > 0:
            logger.info(f"Resuming: {completed} chunks already processed")
            chunks = chunks[completed:]

        # Потоковая обработка
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", initial=completed, total=len(chunks)+completed)):
            chunk_id = completed + i
            try:
                # ПРАВИЛЬНО: extract_from_chunk возвращает dict, передаём его напрямую
                llm_result = self.llm.extract_from_chunk(chunk['text'], {
                    'date_from': chunk['date_from'],
                    'date_to': chunk['date_to'],
                    'msg_count': chunk['msg_count'],
                    'authors': chunk['authors']
                })
                builder.process_chunk(chunk_id, chunk, llm_result)
            except Exception as e:
                logger.error(f"Chunk {chunk_id}: {e}")
                cursor = builder.conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_log (chunk_id, timestamp, messages_processed, status)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, datetime.now().isoformat(), len(chunk.get('message_ids', [])), f'error: {str(e)[:100]}'))
                builder.conn.commit()

        builder.print_stats()
        logger.info("Generating Obsidian vault...")
        builder.generate_obsidian_vault()
        builder.close()

        print("┌────────────────────────────────────────┐")
        print("│ DONE!                                  │")
        print("└────────────────────────────────────────┘")

        print(f"Vault: {Path(self.output_dir) / 'obsidian_vault'}")
        print(f"Database: {Path(self.output_dir) / 'personality_graph.db'}")

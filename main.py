"""
Telegram Multi-Personality Graph Builder (Obsidian Vault Generator)
=====================================================================================
Потоковая обработка: каждый чанк сразу коммитится в БД.
Строит профили для ВСЕХ упомянутых в переписке людей.
Генерирует иерархическое дерево досье для каждого человека.

Usage:
    ollama serve
    python main.py --input chat_export.json --output ./vault --model qwen3:1.7b

Requirements:
    pip install requests tqdm
"""

import logging
import argparse

from modules.Pipeline import Pipeline


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Multi-personality graph from Telegram (Ollama)")
    parser.add_argument("--input", "-i", required=True, help="Telegram JSON export")
    parser.add_argument("--output", "-o", default="./vault", help="Output directory")
    parser.add_argument("--ollama-url", "-u", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", "-m", default="qwen3:1.7b", help="Model name")
    parser.add_argument("--chunk-size", "-c", type=int, default=None, help="Chunk size (auto)")
    parser.add_argument("--target-user", "-t", default=None, help="Not used in multi-mode")

    args = parser.parse_args()

    pipeline = Pipeline(
        input_file=args.input,
        output_dir=args.output,
        ollama_url=args.ollama_url,
        model=args.model,
        max_chunk_size=args.chunk_size
    )
    pipeline.run()


if __name__ == "__main__":
    main()

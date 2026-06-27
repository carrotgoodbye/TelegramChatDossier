"""
Telegram Multi-Personality Graph Builder (Obsidian Vault Generator)
=====================================================================================
Потоковая обработка: каждый чанк сразу коммитится в БД.
Строит профили для ВСЕХ упомянутых в переписке людей.
Генерирует иерархическое дерево досье для каждого человека.

Usage:
    ollama serve
    python main.py --input chat_export.json --output ./vault --model qwen3:1.7b --post-model qwen3:14b

Requirements:
    pip install requests tqdm
"""

import logging
import argparse

from modules.Pipeline import Pipeline
from modules.PostProcessor import PostProcessor


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Multi-personality graph from Telegram (Ollama)")
    parser.add_argument("--input", "-i", required=True, help="Telegram JSON export")
    parser.add_argument("--output", "-o", default="./vault", help="Output directory")
    parser.add_argument("--ollama-url", "-u", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", "-m", default="qwen3:1.7b", help="Model for extraction")
    parser.add_argument("--post-model", "-pm", default=None, help="Model for post-processing (defaults to --model)")
    parser.add_argument("--chunk-size", "-c", type=int, default=None, help="Chunk size (auto)")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip chunk extraction, run only post-processing")
    parser.add_argument("--skip-post", action="store_true", help="Skip post-processing, generate vault from raw DB")

    args = parser.parse_args()
    post_model = args.post_model or args.model

    # Phase 1: Chunk extraction
    if not args.skip_extraction:
        pipeline = Pipeline(
            input_file=args.input,
            output_dir=args.output,
            ollama_url=args.ollama_url,
            model=args.model,
            max_chunk_size=args.chunk_size
        )
        pipeline.run()
    
    # Phase 2: Post-processing (deduplication, consolidation, cleanup)
    if not args.skip_post:
        post = PostProcessor(
            output_dir=args.output,
            ollama_url=args.ollama_url,
            model=post_model
        )
        post.run()
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Raw DB: {args.output}/personality_graph.db")
    print(f"Clean DB: {args.output}/personality_graph_clean.db")
    print(f"Vault: {args.output}/obsidian_vault")
    print("=" * 60)


if __name__ == "__main__":
    main()

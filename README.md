# TelegramChatDossier

Analyze Telegram exports via local LLMs to build multi-person personality dossiers with cognitive patterns, biographical facts, and social graphs — exported as a structured Obsidian knowledge vault.

## What it does
- Extracts **Big Five traits**, cognitive distortions, facts, and relationships from chat history
- Builds profiles for **every person mentioned** (speakers + people they talk about)
- Generates an **Obsidian vault** with linked Markdown files, Mermaid graphs, and a browsable knowledge graph
- Works **fully offline** via Ollama — no data leaves your machine

## Quick Start

```bash
# 1. Install Ollama and pull a model
ollama pull llama3.1:8b

# 2. Run the analyzer
python psygraph.py \
  --input telegram_export.json \
  --output ./vault \
  --model llama3.1:8b
```

## Output
```
vault/
├── obsidian_vault/
│   ├── 🏠 Home.md              # Network overview
│   ├── People/
│   │   └── Alice/
│   │       ├── 🧠 Profile.md
│   │       ├── 📜 Biography.md
│   │       ├── 🧩 Patterns.md
│   │       └── 🔗 Relations.md
│   └── 🕸️ Social Graph.md
└── personality_graph.db        # SQLite with all extracted data
```

Open `obsidian_vault/` as a folder in [Obsidian](https://obsidian.md/) and press `Ctrl+G` to explore the graph.

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- `pip install requests tqdm`

## Disclaimer
This tool is intended for analyzing your own chats or with explicit consent from all parties. Respect privacy and local laws.

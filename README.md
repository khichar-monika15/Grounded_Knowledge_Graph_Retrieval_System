# Layer10 Grounded Long-Term Memory System

Extracts structured knowledge from the Enron email dataset, deduplicates entities and claims, stores them in a memory graph, and provides retrieval + visualization.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set API key (already in .env)
# OPENAI_API_KEY=<your-truefoundry-key>

# 3. Download & sample corpus (200 emails from 2001)
uv run python download_corpus.py --sample-size 200

# 4. Run full pipeline
uv run python run_pipeline.py --sample-size 200

# 5. Launch UI
uv run streamlit run app.py
```

## Project Structure

```
├── schema.py           # Pydantic models: Evidence, Entity, Claim
├── extraction.py       # LLM-based extraction + prompt builder
├── dedup.py            # 3-level dedup: artifact, entity, claim
├── graph_builder.py    # NetworkX graph + SQLite persistence
├── retrieval.py        # Embedding-based retrieval + context packs
├── app.py              # Streamlit visualization
├── run_pipeline.py     # End-to-end orchestrator
├── download_corpus.py  # Corpus sampling from Enron CSV
├── config.py           # API keys, paths, model config
├── tests/              # 60 TDD tests (all passing)
├── dataset/            # Enron_emails.csv (~918MB, gitignored)
├── data/               # Sampled emails (generated, gitignored)
└── outputs/            # Graph, DB, context packs (generated)
```

## Running Tests

```bash
uv run pytest tests/ -v
# 60 tests, all pass
```

## Configuration

`config.py` / `.env`:
- `OPENAI_API_KEY` — TrueFoundry JWT token
- `BASE_URL` — `https://gateway.truefoundry.ai`
- `MODEL` — `anthropic/claude-haiku-4-5-20251001`

## Pipeline Flags

```bash
uv run python run_pipeline.py --sample-size 500   # scale up
uv run python run_pipeline.py --skip-download     # reuse existing data/enron_sample.csv
```

## Dataset

The Enron CSV must be at `dataset/Enron_emails.csv`. Download from:
https://www.kaggle.com/datasets/tarunkashyap/enron-clean

See `write_up.md` for full design documentation.

from dotenv import load_dotenv
import os

load_dotenv()

# LLM provider config — used by litellm in extraction.py
# To switch to Ollama: set MODEL="ollama/llama3", BASE_URL="http://localhost:11434",
# and leave OPENAI_API_KEY empty (Ollama doesn't need a key).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://gateway.truefoundry.ai"
MODEL = "anthropic/claude-haiku-4-5-20251001"

DB_PATH = "outputs/memory.db"
KUZU_DB_PATH = "outputs/kuzu_db"
EXTRACTIONS_DIR = "outputs/extractions"
CONTEXT_PACKS_DIR = "outputs/context_packs"
SAMPLE_DATA_PATH = "data/enron_sample.csv"
GRAPH_JSON_PATH = "outputs/graph.json"

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://gateway.truefoundry.ai"
MODEL = "anthropic/claude-haiku-4-5-20251001"
DB_PATH = "outputs/memory.db"
EXTRACTIONS_DIR = "outputs/extractions"
CONTEXT_PACKS_DIR = "outputs/context_packs"
SAMPLE_DATA_PATH = "data/enron_sample.csv"
GRAPH_JSON_PATH = "outputs/graph.json"

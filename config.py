import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "openai/gpt-oss-20b"

VECTORSTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTORSTORE_DIR, "metadata.pkl")

TOP_K = 4

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
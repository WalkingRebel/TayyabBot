import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

VECTORSTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTORSTORE_DIR, "metadata.pkl")

TOP_K = 4

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
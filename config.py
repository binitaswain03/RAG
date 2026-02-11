import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Provider Settings
# Options: "openai", "google"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K = 3

# Verification Settings
CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 2
API_MAX_RETRIES = 5
API_BACKOFF_SECONDS = 5

# Paths
VECTOR_STORE_PATH = "faiss_index"

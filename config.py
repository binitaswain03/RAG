import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K = 3

# Verification Settings
CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 2

# Paths
VECTOR_STORE_PATH = "faiss_index"

import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

import config

print(f"Config API_MAX_RETRIES: {getattr(config, 'API_MAX_RETRIES', 'Not Found')}")
print(f"Config API_BACKOFF_SECONDS: {getattr(config, 'API_BACKOFF_SECONDS', 'Not Found')}")

try:
    from modules.retrieval import Retriever
    print("Retriever module imported successfully.")
except Exception as e:
    print(f"Error importing Retriever: {e}")

try:
    from modules.correction import CorrectionModule
    print("CorrectionModule imported successfully.")
except Exception as e:
    print(f"Error importing CorrectionModule: {e}")

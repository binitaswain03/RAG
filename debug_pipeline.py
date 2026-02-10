import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

import config
from modules.retrieval import Retriever
from modules.correction_v2 import CorrectionModule

def test_pipeline():
    print("--- Starting Debug Pipeline ---")
    
    # Check API Key
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set in config.")
        return
    else:
        print(f"API Key present: {config.OPENAI_API_KEY[:5]}...{config.OPENAI_API_KEY[-3:]}")

    # 1. Test Retriever
    print("\n[1] Initializing Retriever...")
    try:
        retriever = Retriever()
        print("Retriever initialized.")
        
        query = "test query"
        print(f"Searching for: '{query}'")
        chunks = retriever.get_relevant_chunks(query)
        print(f"Retrieved {len(chunks)} chunks.")
    except Exception as e:
        print(f"ERROR in Retriever: {e}")
        chunks = []

    # 2. Test Correction Module (if retriever failed or just to test it)
    print("\n[2] Initializing Correction Module...")
    try:
        corrector = CorrectionModule()
        print("CorrectionModule initialized.")
        
        print("Attempting to reformulate query...")
        reformulated = corrector.reformulate_query("test query", "No context found")
        print(f"Reformulated Query: {reformulated}")
    except Exception as e:
        print(f"ERROR in Correction Module: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()

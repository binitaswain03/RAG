from langchain_community.vectorstores import FAISS
import config
import os
from utils.llm_utils import retry_with_backoff
from utils.llm_factory import get_embeddings

class Retriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store_path = config.VECTOR_STORE_PATH
        self.vector_store = None
        self.reload_vector_store()

    def reload_vector_store(self):
        """Reloads the vector store from disk."""
        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )

    @retry_with_backoff(retries=config.API_MAX_RETRIES, backoff_in_seconds=config.API_BACKOFF_SECONDS)
    def get_relevant_chunks(self, query):
        """Retrieves relevant chunks for a query."""
        if not self.vector_store:
            self.reload_vector_store()
            
        if not self.vector_store:
            return []
            
        return self.vector_store.similarity_search(query, k=config.TOP_K)

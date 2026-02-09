import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

from utils.llm_utils import retry_with_backoff

class DocumentIngestor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
        self.vector_store_path = config.VECTOR_STORE_PATH

    def load_pdf(self, file_path):
        """Loads a PDF file and returns a list of documents."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def process_documents(self, documents):
        """Splits documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        return texts

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def create_vector_store(self, texts):
        """Creates and saves a FAISS vector store from text chunks."""
        vector_store = FAISS.from_documents(texts, self.embeddings)
        vector_store.save_local(self.vector_store_path)
        return vector_store

    def load_vector_store(self):
        """Loads the FAISS vector store from disk."""
        if os.path.exists(self.vector_store_path):
            return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

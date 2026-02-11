import config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import Google GenAI only if needed, or handle import error
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

def get_llm(temperature=0.7):
    """Factory function to get the LLM based on configuration."""
    if config.LLM_PROVIDER == "google":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature
        )
    else:
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=config.OPENAI_API_KEY,
            temperature=temperature
        )

def get_embeddings():
    """Factory function to get the embeddings based on configuration."""
    if config.EMBEDDING_PROVIDER == "google":
        if GoogleGenerativeAIEmbeddings is None:
            raise ImportError("langchain-google-genai is not installed.")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY
        )
    else:
        return OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY
        )

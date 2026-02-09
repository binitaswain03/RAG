# Self-Correcting RAG Pipeline

A production-ready Retrieval Augmented Generation (RAG) system with self-correction capabilities.
It uses LangChain, FAISS, and OpenAI to answer questions from uploaded PDFs, verify the answers, and automatically reformulate queries if the confidence is low.

## Features
- **PDF Ingestion**: Upload and process PDF documents.
- **RAG Pipeline**: Retrieve relevant context and generate answers.
- **Verification**: Checks if the answer is grounded in the documents (Hallucination detection).
- **Self-Correction**: Automatically reformulates the query if the answer is not supported by the context.
- **Streamlit UI**: Simple and interactive web interface.

## Setup

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API Key:
        ```env
        OPENAI_API_KEY=sk-your-api-key-here
        ```

## Running the Application

```bash
streamlit run app.py
```

## Configuration

You can adjust settings in `config.py`:
- `CONFIDENCE_THRESHOLD`: Score below which correction is triggered (default 0.75).
- `MAX_RETRIES`: Number of self-correction attempts (default 2).

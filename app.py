import streamlit as st
import os
import shutil
import importlib
import modules.correction_v2
import modules.retrieval
import importlib
importlib.reload(modules.correction_v2) 

from modules.ingestion import DocumentIngestor
from modules.correction_v2 import PipelineOrchestrator
import config

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")

st.title("Self-Correcting RAG Pipeline")

# Initialize modules (ensure they are reloaded)
if 'ingestor' not in st.session_state:
    st.session_state.ingestor = DocumentIngestor()
if 'rag_pipeline_v3' not in st.session_state:
    st.session_state.rag_pipeline_v3 = PipelineOrchestrator()

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing Document..."):
                # Save uploaded file temporarily
                temp_path = "temp_uploaded.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                try:
                    docs = st.session_state.ingestor.load_pdf(temp_path)
                    chunks = st.session_state.ingestor.process_documents(docs)
                    st.session_state.ingestor.create_vector_store(chunks)
                    st.success(f"Ingested {len(docs)} pages into {len(chunks)} chunks.")
                    
                    # Force reload of retriever
                    st.session_state.rag_pipeline_v3.retriever.reload_vector_store()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

st.header("Ask a Question")
query = st.text_input("Enter your query:")

if query:
    if st.button("Generate Answer"):
        with st.spinner("Running Pipeline..."):
            result = st.session_state.rag_pipeline_v3.run_pipeline(query)
            
            if result:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Answer")
                    st.markdown(result["answer"])
                    
                    if "confidence" in result:
                        st.info(f"Confidence Score: {result['confidence']}")
                        st.text(f"Reasoning: {result.get('reasoning', '')}")
                
                with col2:
                    st.subheader("Pipeline Logs")
                    for log in result.get("logs", []):
                        st.text(f"- {log}")
                    
                    st.subheader("Source Chunks")
                    for i, chunk in enumerate(result.get("chunks", [])):
                        with st.expander(f"Chunk {i+1} (Source: {chunk.metadata.get('source', 'Unknown')})"):
                            st.write(chunk.page_content)

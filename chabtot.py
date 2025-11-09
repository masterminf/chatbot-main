import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import re
import pandas as pd

# Gemini client
from google import genai

# Embeddings - use Gemini for embeddings instead
import numpy as np

# Vector DB
import chromadb

# -------------------------------
# Gemini OCR Parser Function
# -------------------------------
def parse_document_with_gemini(file_path: str, document_type: str) -> dict:
    """Parse a financial document using Gemini API and return structured data."""
    st.info(f"Processing: {file_path} (Type: {document_type})")

    client = genai.Client(api_key='AIzaSyA4El7po54uPeJfbfZVDOtlV1h8Syn8m3s')

    prompt = """
    You are an expert financial document analyzer. Extract ALL information from this document.
    Return it **strictly as valid JSON** without any extra text, explanation, or formatting.
    """

    file_extension = Path(file_path).suffix.lower()

    try:
        if file_extension == ".xlsx":
            excel_data = pd.read_excel(file_path, sheet_name=None)
            file_content = ""
            for sheet_name, sheet_data in excel_data.items():
                file_content += f"Sheet: {sheet_name}\n"
                file_content += sheet_data.to_csv(index=False)
            
            # For Excel, encode as bytes
            file_bytes = file_content.encode('utf-8')
            mime_type = "text/plain"  # Send as text since we've converted to CSV format
        else:
            file_bytes = Path(file_path).read_bytes()
            if file_extension == ".pdf":
                mime_type = "application/pdf"
            elif file_extension == ".docx":
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif file_extension in [".png", ".jpg", ".jpeg"]:
                mime_type = f"image/{file_extension[1:]}" if file_extension != ".jpg" else "image/jpeg"
            else:
                mime_type = "application/octet-stream"

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                genai.types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
        )

        response_text = response.text

        # Clean JSON
        response_text = re.sub(r"```(json)?", "", response_text, flags=re.IGNORECASE).strip()

        # Remove any leading/trailing non-JSON text
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            response_text = response_text[first_brace:last_brace+1]
        
        parsed_data = json.loads(response_text)

        parsed_data["document_type"] = document_type
        parsed_data["source_file"] = Path(file_path).name
        parsed_data["processing_timestamp"] = datetime.now().isoformat()

        return parsed_data

    except Exception as e:
        st.error(f"Error parsing document: {e}")
        return {"error": str(e)}


# ---------------------------
# Embeddings (Using Gemini)
# ---------------------------
st.cache_data(ttl=3600)
def get_embedding(text: str) -> list:
    """Generate embeddings using Gemini API."""
    try:
        client_emb = genai.Client(api_key='AIzaSyA4El7po54uPeJfbfZVDOtlV1h8Syn8m3s')
        result = client_emb.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return [0.0] * 768

# ---------------------------
# Chroma client and collection (persistent)
# ---------------------------
PERSIST_DIR = "chromadb_persist"

@st.cache_resource
def get_chroma_client_and_collection(collection_name: str = "documents"):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # create or get collection
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"created": datetime.now().isoformat()}
        )
    except Exception as e:
        st.error(f"Error getting collection: {e}")
        collection = client.get_collection(name=collection_name)
    
    return client, collection

client, collection = get_chroma_client_and_collection()

# ---------------------------
# Utilities: JSON -> text, chunking
# ---------------------------
def json_to_text(parsed_data: dict, exclude_meta: bool = True) -> str:
    metadata_keys = ["processing_timestamp", "source_file", "document_type"]
    content_data = {k: v for k, v in parsed_data.items() if not (exclude_meta and k in metadata_keys)}
    # Convert to pretty JSON text
    return json.dumps(content_data, indent=2, ensure_ascii=False)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

# ---------------------------
# Upsert parsed document into Chroma
# ---------------------------
def add_parsed_doc_to_chroma(parsed_data: dict, collection, chunk_size=800):
    text = json_to_text(parsed_data)
    chunks = chunk_text(text, chunk_size=chunk_size)
    ids = []
    docs = []
    metadatas = []
    embeddings = []
    base_id = parsed_data.get("source_file", str(uuid.uuid4()))
    
    for idx, chunk in enumerate(chunks):
        doc_id = f"{base_id}__chunk__{idx}"
        ids.append(doc_id)
        docs.append(chunk)
        metadatas.append({
            "source_file": parsed_data.get("source_file", "unknown"),
            "document_type": parsed_data.get("document_type", "unknown"),
            "chunk_index": idx,
            "processing_timestamp": parsed_data.get("processing_timestamp", "")
        })
        embeddings.append(get_embedding(chunk))
    
    # Upsert documents
    collection.upsert(
        documents=docs,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

# ---------------------------
# Retrieval
# ---------------------------
def retrieve_top_k(query: str, collection, k: int = 4):
    q_emb = get_embedding(query)
    # Chroma's query interface
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    # results: dict with lists
    out = []
    if results and len(results["documents"]) > 0:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            out.append({"document": doc, "metadata": meta, "distance": dist})
    return out

# ---------------------------
# RAG answer via Gemini
# ---------------------------
def answer_with_gemini(retrieved: list, user_query: str) -> str:
    client_g = genai.Client(api_key='AIzaSyA4El7po54uPeJfbfZVDOtlV1h8Syn8m3s')
    # Build context
    context_parts = []
    for r in retrieved:
        src = r["metadata"].get("source_file", "unknown")
        idx = r["metadata"].get("chunk_index", -1)
        context_parts.append(f"Source: {src} (chunk {idx})\n{r['document']}\n---")
    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful, concise assistant that answers user questions using ONLY the provided document context.
If the answer is not contained in the context, say "I don't know — the document doesn't provide enough information."

Context:
{context}

Question:
{user_query}

Answer concisely and clearly. If you reference a fact, mention the source file (source filename).
Return plain text (no JSON or code fences).
"""
    response = client_g.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[prompt],
    )
    return response.text

# ---------------------------
# Streamlit UI
# ---------------------------
#st.set_page_config(page_title="RAG Chatbot — Gemini + Chroma", layout="wide")
st.title("📄 RAG Chatbot — Gemini (OCR) + Chroma (vector store)")

st.sidebar.header("Configuration")
# st.sidebar.write("Add your Gemini API key in Streamlit secrets (.streamlit/secrets.toml):")
# st.sidebar.code('GEMINI_API_KEY = "YOUR_API_KEY"')

uploaded_file = st.file_uploader("Upload document (pdf/docx/xlsx/png/jpg)", type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg"])

if uploaded_file:
    doc_type = st.selectbox("Document type", ["general", "invoice", "ledger", "certificate", "stock"], index=0)
    if st.button("Process & Index"):
        # save temporary file
        tmp_path = Path(f"./temp_{uploaded_file.name}")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Parsing with Gemini..."):
            parsed = parse_document_with_gemini(str(tmp_path), doc_type)

        if parsed.get("error"):
            st.error(f"Parsing error: {parsed['error']}")
        else:
            st.success("Parsed successfully — indexing to Chroma.")
            add_parsed_doc_to_chroma(parsed, collection)
            st.json(parsed)

        tmp_path.unlink(missing_ok=True)

st.markdown("---")
st.subheader("Chat (RAG) with uploaded documents")
query = st.text_input("Enter your question about your documents:")

top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=1, max_value=8, value=4)

if st.button("Ask") and query:
    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve_top_k(query, collection, k=top_k)

    if not retrieved:
        st.warning("No documents in the database yet. Upload & process a document first.")
    else:
        st.write("**Retrieved context (top results):**")
        for i, r in enumerate(retrieved):
            st.markdown(f"**Result {i+1}** — source: {r['metadata'].get('source_file')} (chunk {r['metadata'].get('chunk_index')}) — distance: {r['distance']:.4f}")
            st.write(r['document'][:800] + ("..." if len(r['document']) > 800 else ""))

        with st.spinner("Asking Gemini for an answer..."):
            answer = answer_with_gemini(retrieved, query)

        st.markdown("### ✅ Answer")
        st.write(answer)

# Footer: quick actions
st.sidebar.markdown("---")
if st.sidebar.button("List indexed documents"):
    # naive listing by source_file dedup from metadata
    try:
        all_docs = collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        all_sources = set()
        for meta in metadatas:
            if meta and "source_file" in meta:
                all_sources.add(meta["source_file"])
        if all_sources:
            st.sidebar.write(list(all_sources))
        else:
            st.sidebar.write("No documents indexed yet.")
    except Exception as e:
        st.sidebar.write(f"Error listing docs: {e}")
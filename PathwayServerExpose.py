import os
from datetime import datetime
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# === CONFIGURATION ===
DATA_DIR = "./Data/CleanedPDFsSummarized"
CACHE_DIR = "./Cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Or use 'bge-large-en-v1.5' if you have more RAM
HOST, PORT = "127.0.0.1", 8666

# === 1. Load data from filesystem in streaming mode (with metadata) ===
data = pw.io.fs.read(
    DATA_DIR,
    format="plaintext_by_file",
    mode="streaming",
    with_metadata=True,
)



# === 2. Add filename to metadata ===
def extract_filename(metadata):
    return os.path.basename(metadata["path"])

data_with_filename = data.with_columns(
    file_name=pw.apply(lambda m: extract_filename(m), data._metadata)
)

# === 3. Define embedding model ===
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

# === 4. Chunking & Metadata enrichment ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

def enrich_chunk_metadata(chunk: Document, current_metadata: dict) -> dict:
    """Attach additional metadata per chunk."""
    preview = chunk.page_content[:50] + "..." if len(chunk.page_content) > 50 else chunk.page_content
    return {
        **current_metadata,
        "chunk_length": len(chunk.page_content),
        "text_content_preview": preview,
        "file_path": current_metadata.get("file_name", "unknown_file"),
    }

def split_and_annotate(text: str, file_name: str):
    """Split text into chunks and enrich each chunk's metadata."""
    chunks = splitter.split_text(text)
    result = []
    for idx, chunk_text in enumerate(chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={"file_name": file_name, "chunk_id": idx}
        )
        enriched = enrich_chunk_metadata(doc, doc.metadata)
        result.append({"page_content": chunk_text, "metadata": enriched})
    return result

data_with_chunks = data_with_filename.with_columns(
    chunks=pw.apply(split_and_annotate, pw.this.data, pw.this.file_name)
)

# === 5. Run Vector Store Server ===
server = VectorStoreServer.from_langchain_components(
    data,
    embedder=embeddings,
    splitter=splitter,
)

server.run_server(
    host=HOST,
    port=PORT,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem(CACHE_DIR),
)

print(f"✅ Pathway Vector Store Server running on {HOST}:{PORT}")
print(f"📂 Monitoring directory: {DATA_DIR}")
print(f"🧠 Embedding model: {EMBEDDING_MODEL}")

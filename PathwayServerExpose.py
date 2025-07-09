import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime
import os
import re

# === CONFIGURATION ===
DATA_DIR = "./Data/CleanedPDFsSummarized"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Or use 'bge-large-en-v1.5' for higher accuracy (needs more RAM)
CACHE_DIR = "./Cache"
HOST = "127.0.0.1"
PORT = 8666

# === 1. Load data from filesystem in streaming mode (with metadata) ===
data = pw.io.fs.read(
    DATA_DIR,
    format="plaintext",
    mode="streaming",
    with_metadata=True,
)

def create_base_metadata(metadata):
    filename = metadata['path']
    url_path = str(filename).split("/")[-1]  # Default    
    return {"file_name": url_path}

# After this the data_with_custom_metadata has (data: str, file_name: str, _metadata: pw.json("path":, "created_at":, "modified_at":, "seen_at": ,"size": ))
data_with_custom_metadata = data.with_columns(file_name=pw.apply(lambda metadata: str(create_base_metadata(metadata)['file_name']), data._metadata))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

def custom_metadata_extractor(chunk, current_metadata):
    
    file_name = current_metadata.get("file_name", "unknown_file")
    
    return {
        **current_metadata, # Inherit all base_metadata from the file
        "chunk_length": len(chunk.page_content), # Length of the actual chunk text
        "text_content_preview": chunk.page_content[:50] + "..." if len(chunk.page_content) > 50 else chunk.page_content, # Small preview
        "file_path": file_name
    }

def chunk_and_annotate(text, file_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    docs = []
    for idx, chunk_text in enumerate(chunks):
        doc = Document(page_content=chunk_text, metadata={"file_name": file_name, "chunk_id": idx})
        enriched_meta = custom_metadata_extractor(doc, doc.metadata)
        docs.append({"page_content": chunk_text, "metadata": enriched_meta})
    return docs  # This is a list!

data_with_chunks = data_with_custom_metadata.with_columns(chunks=pw.apply(chunk_and_annotate, pw.this.data, pw.this.file_name))

# class PrintObserver(pw.io.python.ConnectorObserver):
#     def on_change(self, key, row, time, is_addition):
#         print("Row:", row)
#         print("Time:", time)
#         print("Is addition:", is_addition)
#         print("---")

#     def on_end(self):
#         print("End of stream.")

# pw.io.python.write(data_with_chunks, PrintObserver())
# pw.run()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]  # Prioritize paragraph/sentence boundaries
)

server = VectorStoreServer.from_langchain_components(
    data,
    embedder=embeddings,
    splitter=splitter,
)

server.run_server(
    host=HOST,
    port=PORT,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem(CACHE_DIR)
)

print(f"Pathway Vector Store Server running on {HOST}:{PORT}")
print(f"Monitoring directory: {DATA_DIR}")
print(f"Embedding model: {EMBEDDING_MODEL}")
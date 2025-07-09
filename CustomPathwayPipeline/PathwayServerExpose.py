import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime
import os
import re
from config import DATA_DIR, EMBEDDING_MODEL, CACHE_DIR, PATHWAY_HOST, PATHWAY_PORT,SERVER_LOGGING_DIR
import logging
logging.basicConfig(filename=SERVER_LOGGING_DIR, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# === 1. Load data from filesystem in streaming mode (with metadata) ===
data = pw.io.fs.read(
    DATA_DIR,
    format="plaintext",
    mode="streaming",
    with_metadata=True,
)

# After this the data_with_custom_metadata has (data: str, file_name: str, _metadata: pw.json("path":, "created_at":, "modified_at":, "seen_at": ,"size": ))
@pw.udf
def augment_metadata(metadata: pw.Json) -> dict:
    metadata = metadata.as_dict()
    metadata["filename"] = metadata["path"].split("/")[-1]
    del metadata["path"]
    return metadata

data = data.with_columns(_metadata=augment_metadata(pw.this._metadata))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

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
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem(CACHE_DIR)
)

print(f"Pathway Vector Store Server running on {PATHWAY_HOST}:{PATHWAY_PORT}")
print(f"Monitoring directory: {DATA_DIR}")
print(f"Embedding model: {EMBEDDING_MODEL}")
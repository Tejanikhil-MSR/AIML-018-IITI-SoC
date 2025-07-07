import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
DATA_DIR = "/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data/Summarized_PDFs"
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

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

#from langchain.text_splitter import MarkdownHeaderTextSplitter, NLTKTextSplitter

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

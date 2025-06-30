import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

# Read your text documents (streaming mode, with metadata)
data = pw.io.fs.read(
    "/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data/",
    format="plaintext",
    mode="streaming",
    with_metadata=True,
)

# Use a HuggingFace model for embeddings 
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bge-large-en-v1.5")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# Chunk the text using a character-based splitter
# splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[ ".", " ", ""],  # prefers splitting on paragraphs, then lines, etc.
)

# Set host and port for the webserver
host = "127.0.0.1"
port = 8666

# Start the Pathway VectorStoreServer with your data, splitter, and embedder
server = VectorStoreServer.from_langchain_components(
    data,
    embedder=embeddings,
    splitter=splitter,
)

# Run the server (with caching for efficiency)
server.run_server(
    host=host,
    port=port,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem("./Cache")
)

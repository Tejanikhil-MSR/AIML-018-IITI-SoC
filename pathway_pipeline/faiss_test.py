from embedding import get_embedder
from retrieval import FaissRetriever

# Create embedder and retriever
embedder = get_embedder(device="cpu")
retriever = FaissRetriever(embedder)

# Sample documents
docs = [
    {"text": "Hello world"},
    {"text": "Refund policy is 30 days"},
    {"text": "Contact support for more info"},
]

# Build FAISS index
retriever.build_index(docs)

# Test a sample query
results = retriever.retrieve("refund", k=2)
for i, doc in enumerate(results):
    print(f"Result {i+1}:", doc["text"])

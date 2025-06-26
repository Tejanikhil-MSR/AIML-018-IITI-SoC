from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory, UsearchKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore

def get_retriever_factory(embedder):
    # return BruteForceKnnFactory(embedder=embedder)
    retriever_factory = UsearchKnnFactory(
        embedder=embedder,
        dimensions=768,  # or 768 / 1536 depending on your embedding model
        reserved_space=3000,        # estimated number of documents/chunks
        connectivity=16,            # number of neighbors per node
        expansion_add=128,          # controls index construction time vs. quality
        expansion_search=64,        # controls query speed vs. accuracy
    )
    return retriever_factory

def get_document_store(docs, retriever_factory, parser, splitter):
    return DocumentStore(
        docs=docs,
        retriever_factory=retriever_factory,
        parser=parser,
        splitter=splitter,
    )

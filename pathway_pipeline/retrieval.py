from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore

def get_retriever_factory(embedder):
    return BruteForceKnnFactory(embedder=embedder)

def get_document_store(docs, retriever_factory, parser, splitter):
    return DocumentStore(
        docs=docs,
        retriever_factory=retriever_factory,
        parser=parser,
        splitter=splitter,
    )

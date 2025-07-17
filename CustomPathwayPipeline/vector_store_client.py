from langchain_community.vectorstores import PathwayVectorClient
from langchain_core.runnables import RunnableLambda
from config import PATHWAY_HOST, PATHWAY_PORT, DEFAULT_REF_LINK

class PathwayRetriever:
    """
    Manages connection to Pathway Vector Store and document retrieval.
    """
    def __init__(self):
        self.client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)
        self.retriever = self.client.as_retriever()
        self.extracted_context_chain = self.retriever | RunnableLambda(self._format_retrieved_docs)

    def _format_retrieved_docs(self, docs: list) -> dict:
        """
        Formats retrieved documents into a dictionary with context and reference links.
        This function is similar to your original `format_docs`.
        """
        formatted_context = []
        reference_links = []
        keywords = []
        # Limiting to top 3 documents as per your original logic
        for doc in docs[0:3]:
            formatted_context.append(doc.page_content)
            # Accessing 'filename' from metadata as per your Pathway server setup
            reference_links.append(doc.metadata.get("filename", DEFAULT_REF_LINK))
            keywords.append(doc.metadata.get("keywords", " "))
               
        return {
            "context": "\n\n".join(formatted_context),
            "reference_links": "\n".join(reference_links),
            "keywords": "<doc>".join(keywords),
            
        }

    def get_context_and_links(self, query: str) -> dict:
        """
        Invokes the retriever to get relevant context and reference links.
        """
        return self.extracted_context_chain.invoke(query)

pathway_retriever = PathwayRetriever()
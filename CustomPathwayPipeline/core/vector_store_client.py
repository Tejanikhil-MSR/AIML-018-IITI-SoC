from langchain_community.vectorstores import PathwayVectorClient
from langchain_core.runnables import RunnableLambda

class PathwayRetriever:
    """
    Manages connection to Pathway Vector Store and document retrieval.
    Made generic to allow configurable host, port, and number of retrieved documents.
    """
    def __init__(self, host: str, port: int, default_ref_link: str, num_docs_to_return: int = 3): # Added parameter for the limit
        """
        Initializes the PathwayRetriever.

        Args:
            host (str, optional): The host address for the Pathway Vector Store.
                                  Defaults to PATHWAY_HOST from config.
            port (int, optional): The port for the Pathway Vector Store.
                                  Defaults to PATHWAY_PORT from config.
            default_ref_link (str, optional): Default reference link if metadata is missing.
                                              Defaults to DEFAULT_REF_LINK from config.
            num_docs_to_return (int, optional): The maximum number of top documents to retrieve and format.
                                                Defaults to 3.
        """
        self.default_ref_link = default_ref_link
        self.num_docs_to_return = num_docs_to_return
        
        # Initialize PathwayVectorClient with configurable host and port
        self.client = PathwayVectorClient(host=host, port=port)
        self.retriever = self.client.as_retriever()
        
        # The chain now uses the instance's formatting method
        self.extracted_context_chain = self.retriever | RunnableLambda(self._format_retrieved_docs)

    def _format_retrieved_docs(self, docs: list) -> dict:
        """
        Formats retrieved documents into a dictionary with context, reference links, and keywords.
        The number of documents formatted is controlled by `self.num_docs_to_return`.
        """
        formatted_context = []
        reference_links = []
        keywords = []
        
        # Use the configurable limit for the number of documents
        for doc in docs[0:self.num_docs_to_return]:
            formatted_context.append(doc.page_content)
            # Accessing 'filename' from metadata as per your Pathway server setup
            reference_links.append(doc.metadata["filename"])
            keywords.append(doc.metadata["keywords"])

        # print(formatted_context, reference_links, keywords)  # Debugging output to check retrieved links and keywords

        return {
            "context": "\n\n".join(formatted_context),
            "reference_links": " </> ".join(reference_links),
            "keywords_concat": "\n".join(keywords),
        }

    def get_context_and_links(self, query: str) -> dict:
        """
        Invokes the retriever to get relevant context, reference links, and keywords.
        """
        return self.extracted_context_chain.invoke(query)

# Example of how to use it with custom settings
# custom_pathway_retriever = PathwayRetriever(
#     host="127.0.0.1",
#     port=8102, # A different port
#     default_ref_link="https://custom.example.com",
#     num_docs_to_return=5 # Retrieve top 5 documents
# )

# Example of using the custom retriever
# query = "What are the admission requirements for the B.Tech program?"
# results = custom_pathway_retriever.get_context_and_links(query)
# print(results["context"])
# print(results["reference_links"])
# print(results["keywords"])
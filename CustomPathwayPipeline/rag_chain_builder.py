from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from config import PROMPT_TEMPLATE
from vector_store_client import pathway_retriever # Import the instantiated retriever

class RAGChainBuilder:
    """
    Builds the RAG prompt by combining context, chat history, and user query.
    """
    def __init__(self):
        self.augment_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def get_formatted_prompt(self, user_query: str, user_memory: ConversationBufferMemory, label: str = None) -> str:
        """
        Retrieves the documents via the PathwayVectorClient and formats the prompt.
        
        Args:
            user_query (str): Query from the user.
            user_memory (ConversationBufferMemory): Previous chats of the user.
            label (str, optional): An optional classification label to include in the query. Defaults to None.
            
        Returns:
            str: The fully formatted prompt string ready for the LLM.
        """
        # Attach label to the query if provided for retrieval
        query_for_retrieval = f"[{label}] {user_query}" if label else user_query
        
        # Get context and links from the Pathway retriever
        # This will use the labeled query for document retrieval
        docs_result = pathway_retriever.get_context_and_links(query_for_retrieval)

        temp_chain_data = {
            "context": docs_result["context"],
            "question": user_query, # Keep original user query for the LLM's understanding
            "reference_links": docs_result["reference_links"],
            "current_date": datetime.now().strftime("%d/%m/%Y"),
            "chat_history": user_memory.chat_memory.messages
        }

        # IMPORTANT: The LLM's prompt itself might also benefit from the label,
        # but for now, we're only using it for retrieval.
        # If you want the LLM to 'know' the label:
        # temp_chain_data["question"] = f"[{label}] {user_query}" if label else user_query
        
        prompt_value = self.augment_prompt.invoke(temp_chain_data)
        
        return prompt_value.to_string()

# Instantiate the RAG chain builder
rag_builder = RAGChainBuilder()
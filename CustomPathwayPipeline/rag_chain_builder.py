from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from config import PROMPT_TEMPLATE, SYSTEM_PROMPT
from vector_store_client import pathway_retriever

class RAGChainBuilder:
    """
    Builds the RAG prompt by combining context, chat history, and user query.
    """
    def __init__(self):

        system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

        human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)

        self.augment_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

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

        query_for_retrieval = f"[{label}] {user_query}" if label else user_query
        
        docs_result = pathway_retriever.get_context_and_links(query_for_retrieval)

        temp_chain_data = {
            "context": docs_result["context"],
            "question": user_query,
            "reference_links": docs_result["reference_links"],
            "current_date": datetime.now().strftime("%d/%m/%Y"),
            "chat_history": user_memory.chat_memory.messages
        }

        print(docs_result["reference_links"])
        
        prompt_value = self.augment_prompt.invoke(temp_chain_data)
        
        return prompt_value.to_string(), docs_result["reference_links"]

rag_builder = RAGChainBuilder()
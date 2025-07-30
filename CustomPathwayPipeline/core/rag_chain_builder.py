from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from typing import Any


class RAGChainBuilder:
    """
    Builds the RAG prompt by combining context, chat history, and user query.
    Made generic to accept prompt templates and a retriever instance.
    """
    def __init__(self, system_prompt_template: str, human_prompt_template: str, retriever: Any):
        """
        Initializes the RAGChainBuilder.

        Args:
            system_prompt_template (str): The template string for the system message prompt.
                                        Defaults to SYSTEM_PROMPT from config.
            human_prompt_template (str): The template string for the human message prompt (RAG prompt).
                                        Defaults to PROMPT_TEMPLATE from config.
            retriever (any): An object with a 'get_context_and_links' method (e.g., PathwayRetriever instance).
                                Defaults to the global pathway_retriever instance.
        """
        self.retriever = retriever

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

        self.augment_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def get_formatted_prompt(self, user_query: str, user_memory: ConversationBufferMemory, label: str | None = None) -> tuple[str, str, str]:
        """
        Retrieves the documents via the provided retriever and formats the prompt.

        Args:
            user_query (str): Query from the user.
            user_memory (ConversationBufferMemory): Previous chats of the user.
            label (str, optional): An optional classification label to include in the query. Defaults to None.

        Returns:
            tuple[str, str, str]: A tuple containing:
                                  - The fully formatted prompt string ready for the LLM.
                                  - Reference links as a string.
                                  - Keywords as a string.
        """

        query_for_retrieval = f"[{label}] {user_query}" if label else user_query

        docs_result = self.retriever.get_context_and_links(query_for_retrieval)

        temp_chain_data = {
            "context": docs_result["context"],
            "question": user_query,
            "reference_links": docs_result["reference_links"],
            "current_date": datetime.now().strftime("%d/%m/%Y"),
            "chat_history": user_memory.chat_memory.messages
        }

        # print(f"Retrieved Reference Links: {docs_result['reference_links']}")
        # print(f"Retrieved Keywords: {docs_result.get('keywords', 'N/A')}")

        prompt_value = self.augment_prompt.invoke(temp_chain_data)

        return prompt_value.to_string(), docs_result["reference_links"], docs_result.get("keywords", "")


# custom_system_prompt = "You are a helpful assistant."
# custom_human_prompt = "Please answer the question: {question} based on context: {context}"
# class CustomRetriever:
#     def get_context_and_links(self, query):
#         return {"context": "custom context", "reference_links": "custom_link.com", "keywords": "custom, keywords"}
# custom_retriever_instance = CustomRetriever()
#
# custom_rag_builder = RAGChainBuilder(
#     system_prompt_template=custom_system_prompt,
#     human_prompt_template=custom_human_prompt,
#     retriever=custom_retriever_instance
# )
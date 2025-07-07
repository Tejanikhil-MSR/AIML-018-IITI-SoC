from langchain_community.vectorstores import PathwayVectorClient
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint
template = """
You are a helpful assistant for college-related questions.
given the follwing context {context} and the query {question}
answer the question, Respond only based on the context and query, donot use your knowledge
"""

# Connect to your running server
client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever = client.as_retriever()

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Load model locally
pipe = pipeline("text2text-generation", model="google/flan-t5-base",
                device="cuda",  # Use CUDA if available
    max_length=256,              # Output length
    truncation=True)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)



from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# Convert list of documents to a single string
def format_docs(docs):
    print("\n==== Retrieved Context ====\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}\n")
    return "\n\n".join(doc.page_content for doc in docs)

# Wrap the retriever to format its output
formatted_retriever = retriever | RunnableLambda(format_docs)

prompt = ChatPromptTemplate.from_template(template)

# Build the RAG chain
chain = (
    {"context": formatted_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question
response = chain.invoke("who is best coder of iit indore")
print(f" respomse is {response}")

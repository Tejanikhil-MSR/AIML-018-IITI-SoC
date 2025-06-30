from flask import Flask, request, jsonify
from langchain_community.vectorstores import PathwayVectorClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


template = """
You are a helpful assistant for college-related questions.
Given the following context: 
{context}

And the chat history so far:
{chat_history}

Now, answer the current query:
{question}

Only use the context and the chat history to answer. Do not use any prior knowledge or external information. IF you can't answer based on context, say you do not know the answer to query in polite manner.
"""

app = Flask(__name__)
chat_history = []

# Connect to VectorStore
client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever = client.as_retriever()

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Use "cuda" if available
model.eval()

# Generation args
generation_args = {
    "max_new_tokens": 150,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "pad_token_id": tokenizer.eos_token_id
}

def local_llm(prompt) -> str:
    # Ensure prompt is string, not LangChain object
    prompt_text = prompt.to_string()

    formatted_prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()



prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    print("\n==== Retrieved Context ====\n")
    print(docs[0].page_content if docs else "")
    print("\n================= History ===================\n")
    print(chat_history)
    
    return docs[0].page_content if docs else ""

formatted_retriever = retriever | RunnableLambda(format_docs)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def get_chain(chat_history_str):
    return (
        {
            "context": formatted_retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history_str
        }
        | prompt
        | RunnableLambda(local_llm)  # <- replaced llm with custom function
        | StrOutputParser()
    )

@app.route("/", methods=["POST"])
def chat():
    user_message = request.json.get("messages", "")
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400

    chat_history.append(f"User: {user_message}")
    chat_history_str = "\n".join(chat_history[-5:])

    chain = get_chain(chat_history_str)
    response = chain.invoke(user_message)

    chat_history.append(f"Assistant: {response}")
    return jsonify({"response": response})

@app.route("/")
def index():
    return "RAG Chatbot is live"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8011, debug=True)

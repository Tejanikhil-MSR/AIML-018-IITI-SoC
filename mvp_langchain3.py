from flask import Flask, request, jsonify
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import PathwayVectorClient
import torch

app = Flask(__name__)
memory = ConversationBufferMemory(return_messages=True)

# === Model Setup (Phi3, TinyLlama, etc.) ===
MODEL_CHOICE = "mistral"
MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = MODEL_REGISTRY[MODEL_CHOICE]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
model.eval()

generation_args = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache" : False
}

def local_llm(prompt) -> str:
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    print(f" [prompt] = {prompt_text}")
    formatted_prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

# === Prompt ===
template = """
You are a helpful assistant for college-related questions.
Given the following context: 
{context}

And the chat history so far:
{chat_history}

Now, answer the current query:
{question}

Only use the context and the chat history to answer. If you don't know the answer, say so politely.
"""
prompt = ChatPromptTemplate.from_template(template)

# === Vector Store ===
client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever = client.as_retriever()

def format_docs(docs):
    print("Retrieved:", docs[0].page_content if docs else "")
    return docs[0].page_content if docs else ""

formatted_retriever = retriever | RunnableLambda(format_docs)

def get_chain():
    return (
        {
            "context": formatted_retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.chat_memory.messages
        }
        | prompt
        | RunnableLambda(local_llm)
        | StrOutputParser()
    )

@app.route("/", methods=["POST"])
def chat():
    user_message = request.json.get("messages", "")
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400

    memory.chat_memory.add_message(HumanMessage(content=user_message))
    chain = get_chain()
    response = chain.invoke(user_message)
    memory.chat_memory.add_message(AIMessage(content=response))
    print(f'Respopnse = {response}')

    return jsonify({"response": response})

@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8011, debug=True)

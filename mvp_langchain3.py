from flask import Flask, request, jsonify
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import PathwayVectorClient
import torch
from datetime import datetime

DEFAULT_COLLEGE_WEBSITE_LINK = "https://www.iiti.ac.in/"

# === Model Setup (Phi3, TinyLlama, etc.) ===
MODEL_CHOICE = "tinyllama"
MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}

model_name = MODEL_REGISTRY[MODEL_CHOICE]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    DEVICE="cuda"
except Exception as e:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cpu")
    DEVICE="cpu"
    
model.eval()

generation_args = {"max_new_tokens": 200,
                   "do_sample": True,
                   "temperature": 0.7,
                   "top_p": 0.9,
                   "repetition_penalty": 1.1,
                   "pad_token_id": tokenizer.eos_token_id,
                   "use_cache" : False}

def generate_response(prompt) -> str:
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    print(f" [prompt] = {prompt_text}")
    formatted_prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

app = Flask(__name__)
memory = ConversationBufferMemory(return_messages=True)

# === Connect to Pathway server ===
client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever = client.as_retriever()

# === Prompt ===

PromptTemplate = """
You are a AI bot responding to the user for the question - {question} given the context {context} with the previous conversations {chat_history}.

**Additional Info**
- Current Date : {current_date}

**Important Note** : 
- Only use the context and the chat history to answer. If you don't know the answer, say so politely.
- Ask the user to check the College website for more information {reference_links} if you feel that the context is time sensitive.
- Only specify the time/date if the venue/date specified in the context is near to the current date, else ask them to check the website.
"""
AugmentPrompt = ChatPromptTemplate.from_template(PromptTemplate)


def format_docs(docs) -> dict:
    print("\n==== Retrieved Context ====\n")
    formatted_context = []
    ReferenceLinks = []
    for i, doc in enumerate(docs[0:3]):
        print(f"--- Document Chunk {i+1} ---")
        print(f"Content: {doc.page_content}")
        # Now we can print the custom metadata!
        print(f"File Name: {doc.metadata.get('file_name', 'N/A')}")
        print("-" * 30)
        formatted_context.append(doc.page_content)
        ReferenceLinks.append(doc.metadata.get('file_name', DEFAULT_COLLEGE_WEBSITE_LINK))
        
    return {
            "context": "\n\n".join(formatted_context),
            "reference_links": "\n".join(ReferenceLinks), 
    }
    
ExtractedContext = retriever | RunnableLambda(format_docs)

def get_chain():
    return (
        {
            "context": ExtractedContext["context"],
            "question": RunnablePassthrough(),
            "reference_links": ExtractedContext["reference_links"],
            "current_date": datetime.now().strftime("%d/%m/%Y"),
            "chat_history": lambda _: memory.chat_memory.messages
        }
        | AugmentPrompt
        | RunnableLambda(generate_response)
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

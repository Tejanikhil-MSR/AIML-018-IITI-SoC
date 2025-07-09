from flask import Flask, request, jsonify, session
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import PathwayVectorClient # Your Pathway Vector Client import 
import torch
import queue # Using queue module
import threading
import time
import uuid # For unique request IDs
import asyncio # For Future objects
import os
from datetime import datetime

# === Model Setup (Phi3, TinyLlama, etc.) ===
MODEL_CHOICE = "tinyllama"
MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}

DEFAUL_REF_LINK = "www.iiti.ac.in"

model_name = MODEL_REGISTRY[MODEL_CHOICE]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    DEVICE="cuda"
except Exception as e:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cpu")
    DEVICE="cpu"

model.config.pad_token_id = model.config.eos_token_id
    
model.eval()

generation_args = {"max_new_tokens": 200,
                   "do_sample": True,
                   "temperature": 0.7,
                   "top_p": 0.9,
                   "repetition_penalty": 1.1,
                   "pad_token_id": tokenizer.eos_token_id,
                   "use_cache" : True # KV-Cache
                   }

def generate_response(prompt: list[str]) -> list[str]:
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    print(f" [prompt] = {prompt_text}")
    formatted_prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    decoded_responses = []
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    for i,output in enumerate(outputs):
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded.strip()
        decoded_responses.append(response_text)
        
    return decoded_responses

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only_change_this")

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


def format_retrieved_docs(docs: list) -> dict:
    """
    Formats retrieved documents into a dictionary with context and reference links.

    Args:
        docs (list): List of document objects retrieved from PathwayVectorServer. 
                     Each document will have 'page_content' and 'metadata' attributes.

    Returns:
        dict: A dictionary containing:
            - "context" (str): Concatenated text content from the first 3 documents.
            - "reference_links" (str): Newline-separated list of document source filenames.
    """
    formatted_context = []
    ReferenceLinks = []
    for i, doc in enumerate(docs[0:3]):
        formatted_context.append(doc.page_content)
        ReferenceLinks.append(doc.metadata.get("filename", DEFAUL_REF_LINK))
        
    return {
            "context": "\n\n".join(formatted_context),
            "reference_links": "\n".join(ReferenceLinks), 
           }
    
ExtractedContext = retriever | RunnableLambda(format_retrieved_docs)

def get_formatted_prompt(user_query: str, user_memory: ConversationBufferMemory) -> str:
    """
    Retrieves the documents via the PathwayVectorCliet
    
    Args:
        user_query (str): Query from the user
        user_memory (ConversationBufferMemory): Previous chats of the user
        
    Returns:
        AugmentedPrompt: `RetrievedDocuments` -> `Formatter()` -> `PromptTemplate()` -> `AugmentedPrompt`
        
    """
    docs_result = ExtractedContext.invoke(user_query) # Pass user_query to retriever

    temp_chain_data = {
        "context": docs_result["context"],
        "question": user_query,
        "reference_links": docs_result["reference_links"],
        "current_date": datetime.now().strftime("%d/%m/%Y"),
        "chat_history": user_memory.chat_memory.messages
    }

    prompt_value = AugmentPrompt.invoke(temp_chain_data)
    
    return prompt_value.to_string()

request_queue = Queue()
MAX_BATCH = 4
BATCH_TIMEOUT = 0.05  # seconds
response_futures = {}  # id -> Future

def batch_processing_loop():
    while True:
        batch = []
        try:
            item = request_queue.get(timeout=BATCH_TIMEOUT)
            batch.append(item)
        except queue.Empty:
            # If queue is empty, wait for a bit and then check again
            time.sleep(0.01) # Small sleep to prevent busy-waiting
            continue # Continue to the next iteration to check for new items
        
        start_time = time.time()
        while len(batch) < MAX_BATCH and (time.time() - start_time) < BATCH_TIMEOUT:
            try:
                item = request_queue.get(timeout=0.001) # Small timeout to avoid blocking
                batch.append(item)
            except queue.Empty:
                break # No more items, process current batch
            
        if not batch:
            continue # Nothing to process, go back to waiting
        
        prompts_to_process = [item['formatted_prompt'] for item in batch]
        request_ids = [item['request_id'] for item in batch]
        user_memories = [item['user_memory'] for item in batch]
        user_messages = [item['user_message'] for item in batch]
        
        try:
            batch_responses = generate_response(prompts_to_process)
            
            # Distribute responses back to their respective futures
            for i, response in enumerate(batch_responses):
                req_id = request_ids[i]
                original_user_message = user_messages[i]
                original_user_memory = user_memories[i]

                # Add to Langchain memory
                original_user_memory.chat_memory.add_message(AIMessage(content=response))
                
                # Set the result on the future
                if req_id in response_futures:
                    response_futures[req_id].set_result(response)
                    print(f" [Batch] Set result for request_id: {req_id}")
                else:
                    print(f" [Batch] Warning: Future not found for request_id: {req_id}")
                    
        except Exception as e:
            print(f" [Batch] Error during batch generation: {e}")
            # If an error occurs, set exception on all futures in the batch
            for i, req_id in enumerate(request_ids):
                if req_id in response_futures:
                    response_futures[req_id].set_exception(e)

# Start the batch processing thread when the app starts
batch_thread = threading.Thread(target=batch_processing_loop, daemon=True)
batch_thread.start()


@app.route("/", methods=["POST"])
def chat():
    user_message = request.json.get("messages", "")
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    if 'chat_history_memory' not in session:
        session['chat_history_memory'] = {}
        
    user_id = session.sid
    
    if user_id not in session['chat_history_memory']:
        session['chat_history_memory'][user_id] = ConversationBufferMemory(return_messages=True)
    
    user_memory = session['chat_history_memory'][user_id]
    
    user_memory.chat_memory.add_message(HumanMessage(content=user_message))
    
    formatted_prompt = get_formatted_prompt(user_message, user_memory)
    
    request_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    # Create a future object = eventual response of an async operation
    future = loop.create_future() # Create an asyncio Future for this request
    response_futures[request_id] = future # Store the future
    
    request_queue.put({'request_id': request_id, 'formatted_prompt': formatted_prompt,
                       'user_memory': user_memory, # Pass the memory object to update after generation
                       'user_message': user_message # Also pass original user message for memory update
                      })
    
    return jsonify({"request_id": request_id, "status": "processing"}), 202 # 202 Accepted

@app.route("/status/<request_id>", methods=["GET"])
def get_status(request_id):
    future = response_futures.get(request_id)
    if not future:
        return jsonify({"status": "not_found", "message": "Request ID not found or already processed"}), 404
    
    if future.done():
        try:
            response = future.result()
            # Clean up the future once done
            del response_futures[request_id]
            return jsonify({"status": "completed", "response": response}), 200
        except Exception as e:
            # Handle exceptions that occurred during batch processing
            del response_futures[request_id]
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "processing"}), 200


@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {MODEL_CHOICE}"

if __name__ == "__main__":
    # Ensure a default event loop is set for `asyncio.create_future` to work
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app.run(host="0.0.0.0", port=8011, debug=True)
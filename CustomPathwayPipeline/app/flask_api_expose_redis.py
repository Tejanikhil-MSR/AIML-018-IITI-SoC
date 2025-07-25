from flask import Flask, request, jsonify, session
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import uuid
import asyncio
import random
import logging
import sys
import redis
from datetime import timedelta
from flask_session import Session # for server side in-memory storage

sys.path.append("../")

from CustomPathwayPipeline.config import config # type: ignore

from CustomPathwayPipeline.core import PathwayRetriever, RAGChainBuilder, LLMModelLoader, QueryClassifier # type: ignore
from CustomPathwayPipeline.app import BatchProcessor # type: ignore

logging.basicConfig(filename=config.DATA.INFO_LOGGING, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.Redis(host='localhost', port=6379, db=0)
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True  # Optional: adds security
app.config['SESSION_KEY_PREFIX'] = 'rag_session:'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)

# Initialize session with app
Session(app)

###### Initializing Module #######

pathway_retriever = PathwayRetriever(
    host=config.PATHWAY.HOST,
    port=config.PATHWAY.PORT,
    default_ref_link=config.PATHWAY.DEFAULT_REF_LINK,
    num_docs_to_return=config.PATHWAY.k
)

rag_builder = RAGChainBuilder(
    system_prompt_template=config.PROMPTS.SYSTEM_PROMPT,
    human_prompt_template=config.PROMPTS.PROMPT_TEMPLATE,
    retriever=pathway_retriever # Inject the initialized pathway_retriever
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    bnb_4bit_compute_dtype=torch.float16
)

response_generator = LLMModelLoader(
    model_name=config.MODEL.MODEL_REGISTRY[config.MODEL.MODEL_CHOICE],
    device=config.MODEL.DEVICE,
    generation_args=config.MODEL.GENERATION_ARGS,
    quantization_config=bnb_config
)

batch_processor = BatchProcessor(
    llm_generator=response_generator, # Inject the initialized response_generator
    max_batch_size=config.BATCHING.MAX_BATCH_SIZE,
    batch_timeout_seconds=config.BATCHING.BATCH_TIMEOUT_SECONDS
)

query_classifier = QueryClassifier()

###################################

def get_user_memory() -> ConversationBufferMemory:
    """
        Retrieves or creates a ConversationBufferMemory for the current session.
        The actual messages are stored in the Flask session.
    """
    if 'chat_messages' not in session:
        session['chat_messages'] = []
    
    memory = ConversationBufferMemory(return_messages=True)
    
    for msg_data in session['chat_messages']:
        if msg_data['type'] == 'human':
            memory.chat_memory.add_message(HumanMessage(content=msg_data['content']))
        elif msg_data['type'] == 'ai':
            memory.chat_memory.add_message(AIMessage(content=msg_data['content']))
            
    return memory

def save_user_memory_to_session(memory_messages: list):
    """
        Saves the provided serializable list of messages to the Flask session.
    """
    session['chat_messages'] = memory_messages

def handle_direct_response(user_query: str, user_memory: ConversationBufferMemory) -> str:
    """
    Generates a direct response for conversational/greeting queries using config settings.
    """
    
    query_lower = user_query.lower()
    # Access conversational labels and responses from the new config structure
    if any(keyword in query_lower for keyword in config.CONVERSATION.GREETING_LABELS):
        response = random.choice(config.CONVERSATION.GREETING_RESPONSES)
    elif any(keyword in query_lower for keyword in config.CONVERSATION.SEND_OFF_LABELS):
        response = random.choice(config.CONVERSATION.SEND_OFF_RESPONSES)
    else:
        # Fallback if no specific greeting/send-off, can also be customized in config
        response = random.choice(config.CONVERSATION.GREETING_RESPONSES) # Re-use greeting responses for generic hello
    
    # Update the memory directly here since it's an immediate response
    user_memory.chat_memory.add_message(AIMessage(content=response))
    save_user_memory_to_session(user_memory.chat_memory.messages) # Save immediately
    
    return response

@app.route("/", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("messages", "") # type: ignore
    selected_label = data.get("selected_label") # type: ignore
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    # Get memory based on current session
    user_memory = get_user_memory()
    
    if selected_label:

        original_message = session.pop('original_user_message', user_message)
        
        user_memory.chat_memory.add_message(HumanMessage(content=original_message))
        
        # rag_builder.get_formatted_prompt now returns prompt, links, and keywords
        formatted_prompt, reference_links, keywords = rag_builder.get_formatted_prompt(original_message, user_memory, label=selected_label)
        
        # Use logging with f-string for better formatting with multiple args
        logging.info(f"Files retrieved : {reference_links}")
        
        request_id = str(uuid.uuid4())
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        future = loop.create_future()
        
        batch_processor.add_request_to_queue({
            'request_id': request_id,
            'formatted_prompt': formatted_prompt,
            'initial_chat_messages': session['chat_messages'], # Pass current messages
            'user_message': original_message,
            'keywords': keywords # Still pass keywords, though BatchProcessor might not use it generically
            
        }, future)
        
        return jsonify({"request_id": request_id, "status": "processing_with_label", "label": selected_label}), 202

    else:

        probable_labels = query_classifier.classify_query(user_message)
        
        if not probable_labels:
            probable_labels = ["General Info"]
            
        session['original_user_message'] = user_message # Store original message for later use for the current session

        return jsonify({ # as an acknwloedgement message
                        "status": "label_selection_needed",
                        "message": "Please select the most relevant category for your query:",
                        "query": user_message,
                        "probable_labels": probable_labels
                       }), 200

@app.route("/status/<request_id>", methods=["GET"]) # providing the cookie here
def get_status(request_id):
    status, result_data = batch_processor.get_future_result(request_id)
    
    if status == "completed":
        response = result_data['response']
        updated_messages = result_data.get('updated_chat_messages')

        if updated_messages is not None:
            save_user_memory_to_session(updated_messages)
            print(f" [App] Saved updated memory for session.")

        return jsonify({"status": status, "response": response}), 200
    elif status == "error":
        return jsonify({"status": status, "message": str(result_data)}), 500 # Ensure error message is string
    else:
        return jsonify({"status": status, "message": result_data if result_data else "Processing..."}), 200

@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {config.MODEL.MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host=config.FLASK_APP.HOST, port=config.FLASK_APP.PORT, debug=True)
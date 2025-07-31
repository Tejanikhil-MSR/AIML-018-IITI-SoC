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

sys.path.append("../")

from CustomPathwayPipeline.config import config # type: ignore

from CustomPathwayPipeline.core import PathwayRetriever, RAGChainBuilder, LLMModelLoader, QueryClassifier # type: ignore
from CustomPathwayPipeline.app import BatchProcessor # type: ignore

logging.basicConfig(filename=config.DATA.INFO_LOGGING, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = config.FLASK_APP.SECRET_KEY # Required for session to work!

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

def load_user_memory_to_buffer_memory() -> ConversationBufferMemory:
    """
        Load the session stored memory into the ConversationBufferMemory.
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
        stores the new messages to the session variable.
        Note : memory messages are in serialized format which means : 
            HumanMessage(content="...") turns into {'type': 'human', 'content': '...'}
            AIMessage(content="...") turns into {'type': 'ai', 'content': '...'}
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
    
    # Donot add greeting/send-off messages to chat history

    return response

def load_multi_hop_queries(k:int)->str:
    """
    Load the last k user requests from the session for multi-hop queries.
    """
    if 'chat_messages' not in session or len(session['chat_messages']) < k:
        return ""
    
    # Get the last k messages, filtering out AI messages
    multi_hop_queries = [msg['content'] for msg in session['chat_messages'][-k:] if msg['type'] == 'human']
    
    return ' <br> '.join(multi_hop_queries)

@app.route("/", methods=["POST"]) # Backend Entrypoint : Will be called from the frontend
def chat():
    data = request.json
    user_message = data.get("messages", "") # type: ignore
    selected_label = data.get("selected_label") # type: ignore
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    # Get memory based on current session
    user_memory = load_user_memory_to_buffer_memory()

    query_lower = user_message.lower().strip()

    # === If label is given the query is already classified - Send it for Document retrieval ===
    if selected_label:

        user_memory.chat_memory.add_message(HumanMessage(content="Previous request : " + user_message))
        multi_hop_user_requests = load_multi_hop_queries(2) + " <br> " + user_message
        formatted_prompt, reference_links, keywords = rag_builder.augment_prompt_with_context(multi_hop_user_requests, user_memory, label=selected_label)
        
        logging.info(f"Prompt to the LLM for generations : {formatted_prompt}")
        logging.info(f"Keywords extracted from the documents : {keywords}")

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
            'user_message': user_message,
            'keywords': keywords, # Still pass keywords, though BatchProcessor might not use it generically
        }, future)
        
        return jsonify({"request_id": request_id, "status": "processing_with_label", "label": selected_label}), 202

    # === Query is not a greeting/send-off and not yet classified ===
    else:
        # === Step1 : Check if the query is a greeting or send-off ===
        greetings = config.CONVERSATION.GREETING_LABELS
        send_offs = config.CONVERSATION.SEND_OFF_LABELS

        if query_lower in [g.lower().strip() for g in greetings + send_offs]:
            print("detected as a greeting or send-off query")
            response = handle_direct_response(user_message, user_memory) 
            return jsonify({"status": "completed", "response": response}), 200


        # === Step2 : If its a normal query then classify the query using QueryClassifier ===
        probable_labels = query_classifier.classify_query(user_message)
        
        # === Step3.0 : If there is no label found, set to "General Info" and proceed directly to retrieval and generation step ===
        if not probable_labels:

            formatted_prompt, reference_links, keywords = rag_builder.augment_prompt_with_context(user_message, user_memory, label="General Info")
            user_memory.chat_memory.add_message(HumanMessage(content="Previous request : " + user_message))
            logging.info(f"Files retrieved : {reference_links}")

            logging.info(f"Prompt to the LLM for generations : {formatted_prompt}")
            logging.info(f"Keywords extracted from the documents : {keywords}")

            # === Step 3.1 : Create a request id for the current request
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
                'initial_chat_messages': session['chat_messages'], 
                'user_message': user_message, 
                'keywords': keywords
            }, future)
            
            return jsonify({"request_id": request_id, "status": "processing_with_label", "label": "General Info"}), 202

        # === Step4 : If probable labels are found, ask user to select one ===
        return jsonify({"status": "label_selection_needed", 
                        "message": "Please select the most relevant category for your query:",
                        "query": user_message, "probable_labels": probable_labels}), 200

@app.route("/status/<request_id>", methods=["GET"])
def get_status(request_id):

    print(request_id)
    status, result_data = batch_processor.get_future_result(request_id)
    
    if status == "completed":
        response = result_data['response']
        updated_messages = result_data.get('updated_chat_messages')

        if updated_messages is not None:
            save_user_memory_to_session(updated_messages)
            print(f" [App] Saved updated memory for session.")

        return jsonify({"status": status, "response": response}), 200
    
    elif status == "error":
        # result_data should be an error message string
        return jsonify({"status": status, "message": str(result_data)}), 500 # Ensure error message is string
    
    else:
        # result_data will be None for the reqeusts being processed
        return jsonify({"status": status, "message": result_data if result_data else "Processing..."}), 200

@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {config.MODEL.MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host=config.FLASK_APP.HOST, port=config.FLASK_APP.PORT, debug=False)
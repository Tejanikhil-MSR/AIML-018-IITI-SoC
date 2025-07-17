from flask import Flask, request, jsonify, session
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import asyncio
import random

from config import FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, MODEL_CHOICE, INFO_LOGGING
from config import GREETING_LABELS, SEND_OFF_LABELS, GREETING_RESPONSES, CONVERSATIONAL_RESPONSES
from rag_chain_builder import rag_builder
from batch_processor import batch_processor
from query_classifier import query_classifier

import logging
logging.basicConfig(filename=INFO_LOGGING, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

def get_user_memory() -> ConversationBufferMemory:
    """
        Retrieves or creates a ConversationBufferMemory for the current session.
        The actual messages are stored in the Flask session.
    """
    # session.permanent = True # Optional
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
    Generates a direct response for conversational/greeting queries.
    This can be a simple rule-based response or a lighter LLM call.
    """
    
    query_lower = user_query.lower()
    if any(keyword in query_lower for keyword in GREETING_LABELS):
        response = random.choice(GREETING_RESPONSES)
    elif any(keyword in query_lower for keyword in SEND_OFF_LABELS):
        response = random.choice(CONVERSATIONAL_RESPONSES)
    else:
        response = "Hello! How can I assist you today?"
    
    # Update the memory directly here since it's an immediate response
    user_memory.chat_memory.add_message(AIMessage(content=response))
    save_user_memory_to_session(user_memory.chat_memory.messages) # Save immediately
    
    return response


@app.route("/", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("messages", "")
    selected_label = data.get("selected_label")
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    # Get memory based on current session
    user_memory = get_user_memory()
    
    if selected_label:

        original_message = session.pop('original_user_message', user_message)
        
        user_memory.chat_memory.add_message(HumanMessage(content=original_message))
        
        formatted_prompt, reference_links, keywords = rag_builder.get_formatted_prompt(original_message, user_memory, label=selected_label)
        
        logging.info("Files retrieved : ", reference_links)
        
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
            'keywords': keywords
            
        }, future)
        
        return jsonify({"request_id": request_id, "status": "processing_with_label", "label": selected_label}), 202

    else:
        # Phase 1: Initial query, classify and ask for user intent
        probable_labels = query_classifier.classify_query(user_message)
        
        if not probable_labels:
            # Default if no specific labels found
            probable_labels = ["General Info"]
            
        session['original_user_message'] = user_message

        return jsonify({
                        "status": "label_selection_needed",
                        "message": "Please select the most relevant category for your query:",
                        "query": user_message,
                        "probable_labels": probable_labels
                       }), 200

@app.route("/status/<request_id>", methods=["GET"])
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
        return jsonify({"status": status, "message": result_data}), 500
    else:
        return jsonify({"status": status, "message": result_data if result_data else "Processing..."}), 200


@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
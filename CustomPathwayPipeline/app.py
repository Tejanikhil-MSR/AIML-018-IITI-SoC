from flask import Flask, request, jsonify, session
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import asyncio # Still needed for create_future

# Import components from our new modules
from config import FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, MODEL_CHOICE
from rag_chain_builder import rag_builder
from batch_processor import batch_processor # We'll modify this to manage shared memory better
from query_classifier import query_classifier

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Helper function to get/create user-specific memory
def get_user_memory() -> ConversationBufferMemory:
    """
    Retrieves or creates a ConversationBufferMemory for the current session.
    The actual messages are stored in the Flask session.
    """
    # session.permanent = True # Optional: Make session last longer than browser close
    if 'chat_messages' not in session:
        session['chat_messages'] = []
    
    # Reconstruct ConversationBufferMemory from session messages
    memory = ConversationBufferMemory(return_messages=True)
    for msg_data in session['chat_messages']:
        if msg_data['type'] == 'human':
            memory.chat_memory.add_message(HumanMessage(content=msg_data['content']))
        elif msg_data['type'] == 'ai':
            memory.chat_memory.add_message(AIMessage(content=msg_data['content']))
    return memory

# Helper function to save user-specific memory
def save_user_memory_to_session(memory_messages: list):
    """
    Saves the provided serializable list of messages to the Flask session.
    """
    session['chat_messages'] = memory_messages


# --- Flask Routes ---

@app.route("/", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("messages", "")
    selected_label = data.get("selected_label")
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    user_memory = get_user_memory() # Get memory based on current session
    
    if selected_label:
        # Phase 2: User has selected a label, proceed with RAG
        original_message = session.pop('original_user_message', user_message)
        
        # Add the original user message to memory
        user_memory.chat_memory.add_message(HumanMessage(content=original_message))
        # No need to save_user_memory_to_session here directly,
        # it will be saved after the LLM response is ready.

        formatted_prompt = rag_builder.get_formatted_prompt(original_message, user_memory, label=selected_label)
        
        request_id = str(uuid.uuid4())
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        future = loop.create_future()
        
        # Pass the current chat_messages and session_id to batch_processor
        # The batch_processor will handle updating the memory and returning
        # the *full updated serializable messages* in the future's result.
        batch_processor.add_request_to_queue({
            'request_id': request_id,
            'formatted_prompt': formatted_prompt,
            'initial_chat_messages': session['chat_messages'], # Pass current messages
            'user_message': original_message
        }, future)
        
        return jsonify({"request_id": request_id, "status": "processing_with_label", "label": selected_label}), 202

    else:
        # Phase 1: Initial query, classify and ask for user intent
        probable_labels = query_classifier.classify_query(user_message)
        
        if not probable_labels:
            probable_labels = ["General Info"] # Default if no specific labels found
            
        session['original_user_message'] = user_message # Store for next step

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
            # Save the updated chat messages back to the Flask session
            save_user_memory_to_session(updated_messages)
            print(f" [App] Saved updated memory for session.")

        return jsonify({"status": status, "response": response}), 200
    elif status == "error":
        return jsonify({"status": status, "message": result_data}), 500
    else: # processing or not_found
        return jsonify({"status": status, "message": result_data if result_data else "Processing..."}), 200


@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

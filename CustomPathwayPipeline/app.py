from flask import Flask, request, jsonify, session
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
import uuid
import asyncio # Still needed for create_future
from query_classifier import query_classifier

# Import components from our new modules
from config import FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, MODEL_CHOICE
from rag_chain_builder import rag_builder
from batch_processor import batch_processor

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY # Use secret key from config

@app.route("/", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("messages", "")
    selected_label = data.get("selected_label") # New field for user's chosen label
    
    if not user_message:
        return jsonify({"error": "Missing 'messages' field"}), 400
    
    # === Multi-user memory management ===
    # Use session to store individual user's chat history memory
    if 'chat_history_memory' not in session:
        session['chat_history_memory'] = {}
        
    user_id = session.sid # Get unique session ID for the user
    
    if user_id not in session['chat_history_memory']:
        session['chat_history_memory'][user_id] = ConversationBufferMemory(return_messages=True)
    
    user_memory = session['chat_history_memory'][user_id]
    
    # Add user's message to their specific chat history
    user_memory.chat_memory.add_message(HumanMessage(content=user_message))
    
    if selected_label:
        # Phase 2: User has selected a label, proceed with RAG
        original_message = session.pop('original_user_message', user_message) # Retrieve original message
        
        # Add the original user message to memory (before RAG response)
        user_memory.chat_memory.add_message(HumanMessage(content=original_message))

        # Generate the full formatted prompt string using the RAG builder, now with the selected label
        formatted_prompt = rag_builder.get_formatted_prompt(original_message, user_memory, label=selected_label)
        
        # === Batching Integration (same as before) ===
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
            'user_memory': user_memory,
            'user_message': original_message # Store original message for memory update after generation
        }, future)
        
        return jsonify({"request_id": request_id, "status": "processing_with_label", "label": selected_label}), 202

    else:
        # Phase 1: Initial query, classify and ask for user intent
        probable_labels = query_classifier.classify_query(user_message)
        
        if not probable_labels:
            # If no labels are found, or only "General Info", directly proceed with RAG (optional)
            # Or you can return a message indicating no specific categories found.
            # For simplicity, if no specific labels found, we'll suggest General Info
            probable_labels = ["General Info"]
            
        # Store the original message in session for the next step
        session['original_user_message'] = user_message

        # Return the labels and instruct the user to select one
        return jsonify({
            "status": "label_selection_needed",
            "message": "Please select the most relevant category for your query:",
            "query": user_message,
            "probable_labels": probable_labels
        }), 200


@app.route("/status/<request_id>", methods=["GET"])
def get_status(request_id):
    # Check the status of the request via the batch processor
    status, result = batch_processor.get_future_result(request_id)
    
    if status == "completed":
        return jsonify({"status": status, "response": result}), 200
    elif status == "error":
        return jsonify({"status": status, "message": result}), 500
    else: # processing or not_found
        return jsonify({"status": status, "message": result if result else "Processing..."}), 200

@app.route("/")
def index():
    return f"RAG Chatbot is live using model: {MODEL_CHOICE}"

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
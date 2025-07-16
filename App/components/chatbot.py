import requests
import streamlit as st
import time # For polling delay
from components.session import add_message # Import to update chat history

# Change this if your backend runs elsewhere
BACKEND_URL = "http://127.0.0.1:8011"
POLLING_INTERVAL_SECONDS = 0.5 # How often Streamlit checks Flask for response

def send_query_to_backend(user_message, selected_label=None):
    """
    Sends the user query to the Flask backend. Handles both initial query
    and label-selected query.
    """
    payload = {"messages": user_message}
    if selected_label:
        payload["selected_label"] = selected_label
    
    try:
        response = requests.post(f"{BACKEND_URL}/", json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
        return None

def poll_backend_status(request_id):
    """
    Polls the Flask backend's status endpoint for the final response.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/status/{request_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to poll status from backend: {e}")
        return None

def get_bot_response_orchestrator(user_input):
    """
    Orchestrates the two-phase communication with the Flask backend.
    """
    # Phase 1: Send initial query for classification
    response_data = send_query_to_backend(user_input)

    if response_data:
        if response_data.get("status") == "label_selection_needed":
            st.session_state.waiting_for_label = True
            st.session_state.original_query_for_label = user_input
            st.session_state.probable_labels = response_data.get("probable_labels", [])
            st.toast("Please select a category.")
            st.rerun() # Rerun to show label buttons
        elif response_data.get("status") == "processing_with_label":
            st.session_state.request_id = response_data["request_id"]
            st.toast("Processing your query...")
            # Do NOT rerun here, let the main app.py polling logic handle it
        else:
            st.error(f"Unexpected response from backend: {response_data}")
            st.session_state.request_id = None
            return "Error: Unexpected backend response."
    else:
        st.session_state.request_id = None
        return "Error: Could not get response from backend."

    # If we reach here, it means a request_id was set for processing.
    # The polling logic will be handled by the main app.py
    return None # Return None here as response is not immediate

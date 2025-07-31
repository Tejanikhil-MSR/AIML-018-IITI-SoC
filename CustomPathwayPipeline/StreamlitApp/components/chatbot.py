import sys
sys.path.append("../")

import requests
import streamlit as st
from frontend_config import BACKEND_URL

def send_query_to_backend(user_message, selected_label=None):
    """
    Sends the user query to the Flask backend. Handles both initial query and label-selected query.
    """
    payload = {"messages": user_message}

    if selected_label:
        payload["selected_label"] = selected_label
    
    try:
        response = requests.post(f"{BACKEND_URL}/", json=payload) # calls the Flask endpoint
        response.raise_for_status()
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
    # Phase 1: Send the user query for classification
    response_data = send_query_to_backend(user_message=user_input, selected_label=None)

    if response_data:

        if response_data.get("status") == "label_selection_needed":
            # prompt user to select a label based on the query classified
            st.session_state.waiting_for_label = True
            st.session_state.current_status="label_selection_needed"
            st.session_state.original_query_for_label = user_input
            st.session_state.probable_labels = response_data.get("probable_labels", [])
            st.toast("Please select a category.")
            st.rerun() # Rerun to show label buttons
        
        # This will run when the function is called from the frontends app.py
        elif response_data.get("status") == "processing_with_label":
            st.session_state.request_id = response_data["request_id"]
            st.session_state.current_status = "processing_with_label"
            st.toast("Processing your query...")
            # Do NOT rerun here, let the main app.py polling logic handle it

        # This is only for handling direct responses like greetings or send-offs
        elif response_data.get("status") == "completed":
            st.session_state.request_id = None # No need to set request_id here, as this is a direct response and doesnt need waiting for backend processing
            st.session_state.current_status = "completed"
            st.session_state.response = response_data["response"]
            st.toast("Processing done...")
            st.rerun() # Rerun to display the final answer

        else:
            st.error(f"Unexpected response from backend: {response_data}")
            st.session_state.request_id = None
            return "Error: Unexpected backend response."
        
    else:
        st.session_state.request_id = None
        return "Error: Could not get response from backend."
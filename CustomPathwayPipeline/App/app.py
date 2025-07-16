import sys
import os
import streamlit as st
import time # For polling delay

# Add components directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components'))

from components.layout import render_header, render_styles
from components.chatbot import get_bot_response_orchestrator, send_query_to_backend, poll_backend_status, POLLING_INTERVAL_SECONDS
from components.session import init_chat_history, add_message, get_chat_history, clear_chat_state

# --- Setup ---
st.set_page_config(page_title="College Chatbot", layout="centered")
render_styles()
render_header()
init_chat_history()

# --- Display Chat Messages ---
# Use a placeholder for the chat history container
chat_history_container = st.container()

with st.container():
    st.markdown("### 💬 Chat History")
    for sender, msg in get_chat_history():
        if sender == "user":
            st.markdown(
                f"""<div style='background-color: #d17a22; color: white; padding: 10px; 
                      border-radius: 10px; margin: 5px 0; max-width: 70%; 
                      margin-left: auto; text-align: right;'>{msg}</div>""",
                unsafe_allow_html=True
            ) # User message bubble (warm orange)
        else: # bot
            st.markdown(
                f"""<div style='background-color: #8d6e63; color: white; padding: 10px; 
                      border-radius: 10px; margin: 5px 0; max-width: 70%; 
                      margin-right: auto; text-align: left;'>{msg}</div>""",
                unsafe_allow_html=True
            ) # Bot message bubble (muted brown)
            
# --- Input Field and Send Button / Label Selection ---

# Check if we are waiting for a label selection
if st.session_state.waiting_for_label:
    st.write("Please select the most relevant category for your query:")
    cols = st.columns(len(st.session_state.probable_labels)) # Create columns for buttons
    
    for i, label in enumerate(st.session_state.probable_labels):
        with cols[i]: # Place button in its column
            if st.button(label, key=f"label_button_{label}"):
                # User selected a label
                st.session_state.waiting_for_label = False
                st.session_state.probable_labels = [] # Clear labels from UI

                # Send the original query with the selected label to backend
                response_data = send_query_to_backend(
                    st.session_state.original_query_for_label,
                    selected_label=label
                )
                
                if response_data and response_data.get("status") == "processing_with_label":
                    st.session_state.request_id = response_data["request_id"]
                    st.toast(f"Processing query with label: {label}...")
                    st.rerun() # Rerun to start polling
                else:
                    st.error("Error initiating RAG processing after label selection.")
                    st.session_state.request_id = None # Clear request_id on error

else: # Not waiting for label, show normal chat input
    user_input = st.text_input("Type your message here:", key="user_input_text")
    if st.button("Send", key="send_button") and user_input.strip():
        # Add user query to chat history immediately for display
        add_message("user", user_input.strip())
        
        # Trigger the backend communication orchestration
        get_bot_response_orchestrator(user_input.strip())
        
        # Clear the input box visually after sending
        st.session_state.user_input_text = "" 
        st.rerun() # Rerun to clear input and display user message

# --- Polling for Response (if a request is in progress) ---
if st.session_state.request_id:
    # Display "Thinking..." message while waiting for backend
    with chat_history_container: # Display inside the chat history area
        st.markdown(
            f"""<div style='background-color: #1565c0; color: white; padding: 10px; 
                  border-radius: 10px; margin: 5px 0; max-width: 70%; 
                  margin-right: auto; text-align: left;'>Thinking...</div>""",
            unsafe_allow_html=True
        )

    status_data = poll_backend_status(st.session_state.request_id)
    
    if status_data:
        if status_data.get("status") == "completed":
            final_response = status_data.get("response")
            if final_response:
                add_message("bot", final_response) # Add bot's final response to history
            st.session_state.request_id = None # Clear request ID to stop polling
            st.rerun() # Rerun to display the final answer and clear "Thinking..."
        elif status_data.get("status") == "error":
            st.error(f"Error from backend: {status_data.get('message', 'Unknown error')}")
            st.session_state.request_id = None # Clear request ID on error
            st.rerun() # Rerun to clear the "Thinking..." message
        else: # status is "processing" or "not_found" (still waiting)
            # Still processing, wait a bit and rerun
            time.sleep(POLLING_INTERVAL_SECONDS)
            st.rerun() # Keep rerunning until completed or error

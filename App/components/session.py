import streamlit as st

def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "request_id" not in st.session_state:
        st.session_state.request_id = None
    if "waiting_for_label" not in st.session_state:
        st.session_state.waiting_for_label = False
    if "original_query_for_label" not in st.session_state:
        st.session_state.original_query_for_label = None
    if "probable_labels" not in st.session_state:
        st.session_state.probable_labels = []

def add_message(sender, text):
    st.session_state.chat_history.append((sender, text))

def get_chat_history():
    return st.session_state.chat_history

def clear_chat_state():
    """Clears all relevant chat-related session states."""
    st.session_state.chat_history = []
    st.session_state.request_id = None
    st.session_state.waiting_for_label = False
    st.session_state.original_query_for_label = None
    st.session_state.probable_labels = []

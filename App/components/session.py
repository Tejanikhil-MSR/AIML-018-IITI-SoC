import streamlit as st

def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_message(sender, text):
    st.session_state.chat_history.append((sender, text))

def get_chat_history():
    return st.session_state.chat_history

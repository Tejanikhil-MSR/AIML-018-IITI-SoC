import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from components.layout import render_header, render_styles
from components.chatbot import get_bot_response
from components.session import init_chat_history, add_message, get_chat_history

# Setup
st.set_page_config(page_title="College Chatbot", layout="centered")
render_styles()
render_header()
init_chat_history()

# Display chat messages inside a container
with st.container():
    st.markdown("### 💬 Chat History")
    for sender, msg in get_chat_history():
        if sender == "user":
            st.markdown(
                f"""<div style='background-color: #2e7d32; color: white; padding: 10px; 
                      border-radius: 10px; margin: 5px 0; max-width: 70%; 
                      margin-left: auto; text-align: right;'>{msg}</div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='background-color: #1565c0; color: white; padding: 10px; 
                      border-radius: 10px; margin: 5px 0; max-width: 70%; 
                      margin-right: auto; text-align: left;'>{msg}</div>""",
                unsafe_allow_html=True
            )

# Input field and send button
user_input = st.text_input("Type your message here:")
if st.button("Send") and user_input.strip():
    add_message("user", user_input.strip())
    bot_reply = get_bot_response(user_input.strip())
    add_message("bot", bot_reply)
    st.rerun()

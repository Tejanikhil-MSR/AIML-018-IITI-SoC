import streamlit as st

def render_styles():
    st.markdown("""
        <style>
        body {
            background-color: #eef2f7;
        }
        .stButton > button {
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("<h1 style='color:#2c3e50;'>🤖 College Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555;'>Ask me anything about your campus!</p>", unsafe_allow_html=True)

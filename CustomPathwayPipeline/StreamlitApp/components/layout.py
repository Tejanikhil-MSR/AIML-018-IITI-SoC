import streamlit as st

def render_styles():
    st.markdown("""
        <style>
        /* General body background */
        body {
            background-color: #f8f0e3; /* A soft, warm off-white */
        }
        
        /* Streamlit main container background */
        .stApp {
            background-color: #f8f0e3; /* Ensure the main app area matches body background */
        }

        /* Customize Streamlit widgets (buttons, text input, etc.) */
        .stButton > button {
            background-color: #d17a22; /* A warm orange for buttons */
            color: white; /* White text on buttons */
            border-radius: 8px; /* Slightly more rounded buttons */
            border: none; /* Remove default border */
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s ease; /* Smooth transition on hover */
        }
        .stButton > button:hover {
            background-color: #e08c3e; /* Lighter orange on hover */
        }

        .stTextInput > div > div > input {
            background-color: #ffffff; /* White background for text input */
            color: #333333; /* Dark gray text color for input */
            border-radius: 8px;
            border: 1px solid #d17a22; /* Orange border for input */
            padding: 10px 15px;
        }

        /* Header styling */
        h1 {
            color: #d17a22; /* Warm orange for the main heading */
            text-align: center;
        }
        p {
            color: #5d4037; /* Darker brown for sub-headings/paragraph text */
            text-align: center;
        }

        /* Chat history heading */
        .stMarkdown h3 {
            color: #8d6e63; /* Muted brown for chat history heading */
            text-align: center;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("<h1 style='color:#d17a22;'>🤖 College Bot</h1>", unsafe_allow_html=True) # Changed color here too
    st.markdown("<p style='color:#5d4037;'>Ask me anything about your campus!</p>", unsafe_allow_html=True) # Changed color here too

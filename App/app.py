import streamlit as st

# Set wide layout for desktop feel
st.set_page_config(
    page_title="Web Interface",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inject custom CSS (theme based on Android dark UI)
st.markdown("""
    <style>
    /* Set background and text color */
    body {
        background-color: #121212;
        color: #FFFFFF;
    }

    /* Style all text elements */
    h1, h2, h3, h4, h5, h6, p {
        color: #FFFFFF;
    }

    /* Customize buttons */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #1565C0;
        transition: 0.3s;
    }

    /* Customize text inputs */
    .stTextInput>div>input {
        background-color: #1E1E1E;
        color: white;
        border-radius: 5px;
        border: 1px solid #333;
        padding: 8px;
    }

    /* Center image/logo if needed */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    /* Adjust columns padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Interface Starts Here ---
st.markdown("<div class='logo-container'><h2>🌐 Web Interface</h2></div>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([1, 3])  # Adjust ratio for sidebar/main content

# Left Sidebar / Info Section
with col1:
    st.subheader("Navigation")
    st.button("Home")
    st.button("Settings")
    st.button("Help")
    st.write("This sidebar mimics a native app drawer.")

# Main Content
with col2:
    st.subheader("Main Interaction Area")

    user_input = st.text_input("Enter your message")
    if st.button("Submit"):
        st.success(f"You entered: {user_input}")

    st.markdown("---")
    st.write("Here you can add charts, tables, or any Streamlit component to mimic Android behavior on a wide web interface.")


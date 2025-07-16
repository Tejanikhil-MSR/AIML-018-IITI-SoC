import os
from datetime import datetime

# --- Model Configuration ---
MODEL_CHOICE = "mistral"

MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}

DEVICE = "cuda" # Default, will try to fall back to CPU

# --- Generation Arguments ---
GENERATION_ARGS = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "use_cache": True # KV-Cache
}

# --- Pathway Vector Store Configuration ---
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8101
DEFAULT_REF_LINK = "www.iiti.ac.in" # Default reference link if metadata is missing

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
Logdir = "./logs"
os.mkdir(f"{Logdir}/{timestamp}")
DEBUGGER_LOGGING = f"./logs/{timestamp}/pathway_server_debug.log"
INFO_LOGGING = f"./logs/{timestamp}/info.log"

DATA_DIR = "../Documentations/"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Or use 'bge-large-en-v1.5' for higher accuracy (needs more RAM)
CACHE_DIR = "./Cache"

# --- Batching Configuration ---
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT_SECONDS = 0.05

# --- Flask App Configuration ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8011
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only_change_this")

# --- System prompt ---
SYSTEM_PROMPT = "You are a friendly and helpful AI assistant for IIT Indore. You answer questions about admissions, academics, campus life, and events. Always be polite and concise."

GREETING_LABELS = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon"]
SEND_OFF_LABELS = ["thank", "thanks", "bye", "goodbye"]

# --- Generation Responses ---
GREETING_RESPONSES = ["Hello! How can I assist you today regarding IIT Indore?", "Hi there! What would you like to know about the college?",
                      "Greetings! I'm here to help with your IIT Indore questions.", "Hey! Ask me anything about IIT Indore.", 
                      "Welcome! How can I guide you about the campus?"]
    
CONVERSATIONAL_RESPONSES = ["You're welcome! Feel free to ask if you have more questions.", "My pleasure! I'm an AI assistant for IIT Indore. How can I help?",
                            "I'm here to provide information about IIT Indore. What's on your mind?", 
                            "It was nice chatting with you! Is there anything else about IIT Indore I can assist with?"]

INTENTS = ["Admissions", "Academics", "Student Life", "Research", "Events"]

# --- RAG Prompt Template ---
PROMPT_TEMPLATE = """
[INST] 
You are a AI bot responding to the user for the question - {question} given the context {context} with the previous conversations {chat_history}.

**Additional Info**
- Current Date : {current_date}
<<SYS>>
**Important Note** : 
- Only use the context and the chat history to answer. If you don't know the answer, say so politely.
- Ask the user to check the College website for more information {reference_links} if you feel that the context is time sensitive.
- Only specify the time/date if the venue/date specified in the context is near to the current date, else ask them to check the website.
<</SYS>>
[/INST]
"""

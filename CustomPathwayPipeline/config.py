import os

# --- Model Configuration ---
MODEL_CHOICE = "phi3"

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
SERVER_LOGGING_DIR = "./logs/pathway_server.log"
DATA_DIR = "../Data/TrialSamples/"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Or use 'bge-large-en-v1.5' for higher accuracy (needs more RAM)
CACHE_DIR = "./Cache"

# --- Batching Configuration ---
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT_SECONDS = 0.05

# --- Flask App Configuration ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8011
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only_change_this")

# --- RAG Prompt Template ---
PROMPT_TEMPLATE = """
You are a AI bot responding to the user for the question - {question} given the context {context} with the previous conversations {chat_history}.

**Additional Info**
- Current Date : {current_date}

**Important Note** : 
- Only use the context and the chat history to answer. If you don't know the answer, say so politely.
- Ask the user to check the College website for more information {reference_links} if you feel that the context is time sensitive.
- Only specify the time/date if the venue/date specified in the context is near to the current date, else ask them to check the website.
"""
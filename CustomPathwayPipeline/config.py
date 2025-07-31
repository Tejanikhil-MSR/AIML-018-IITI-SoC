import os
from datetime import datetime
from dataclasses import dataclass, field # dataclasses are great for configuration groups


@dataclass
class ModelConfig:
    """Configuration for Language Models."""
    MODEL_CHOICE: str = "mistral"
    MODEL_REGISTRY: dict = field(default_factory=lambda: {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "phi3": "microsoft/Phi-3-mini-4k-instruct"
    })
    
    DEVICE: str = "cuda"  # Default, will try to fall back to CPU
    
    GENERATION_ARGS: dict = field(default_factory=lambda: {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "use_cache": True # KVCache
    })

@dataclass
class PathwayConfig:
    """Configuration for Pathway Vector Store."""
    HOST: str = "127.0.0.1"
    PORT: int = 8101
    DEFAULT_REF_LINK: str = "www.iiti.ac.in"  # Default reference link if metadata is missing
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"  # Or use 'bge-large-en-v1.5'
    CACHE_DIR: str = "./Cache"
    k: int = 3

@dataclass
class DataConfig:
    """Configuration for data directories and logging."""
    LOG_DIR: str = "./logs"
    DEBUGGER_LOGGING: str = field(init=False)
    INFO_LOGGING: str = field(init=False)
    
    ROOT_DATA_DIR: str = "./Data/"
    PDF_DATA_DIR: str = field(init=False)
    TEXT_DATA_DIR: str = field(init=False)

    def __post_init__(self):
        # Ensure log directory exists
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.DEBUGGER_LOGGING = os.path.join(self.LOG_DIR, "pathway_server_debug.log")
        self.INFO_LOGGING = os.path.join(self.LOG_DIR, "info.log")

        # Construct full data paths
        self.PDF_DATA_DIR = os.path.join(self.ROOT_DATA_DIR, "PDFFiles")
        self.TEXT_DATA_DIR = os.path.join(self.ROOT_DATA_DIR, "TextFiles")
        # Ensure data directories exist (optional, depends on workflow)
        os.makedirs(self.PDF_DATA_DIR, exist_ok=True)
        os.makedirs(self.TEXT_DATA_DIR, exist_ok=True)

@dataclass
class BatchingConfig:
    """Configuration for request batch processing."""
    MAX_BATCH_SIZE: int = 4
    BATCH_TIMEOUT_SECONDS: float = 0.05

@dataclass
class FlaskAppConfig:
    """Configuration for the Flask application."""
    HOST: str = "0.0.0.0"
    PORT: int = 8011
    # Retrieve secret key from environment variable for security
    SECRET_KEY: str = field(default_factory=lambda: os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only_change_this"))

@dataclass
class PromptsConfig:
    """Configuration for LLM system and RAG prompts."""
    SYSTEM_PROMPT: str = (
        "You are a friendly and helpful AI assistant for IIT Indore. You answer questions about admissions, "
        "academics, campus life, and events. Always be polite and concise."
    )

    SUMMARIZATION_PROMPT : str = """
You are an AI assistant tasked with summarizing tables and rewriting the text content without any noise.
Given a table, convert it into a concise summary that captures the key information. 
Given a text, rewrite the whole text by removing any noise and making it more readable.

For text, respond only with the rewritten text and for the table respond with a concise summary of the table content.
Do not start your message by saying "Here is a summary" or anything like that.

just give me what i have asked for. 

Table or text chunk : {element}
"""

    # Using multiline string for readability of the template
    PROMPT_TEMPLATE: str = f"""
[INST]
You are a AI bot responding to the user for the question - {{question}} given the context {{context}} with the previous conversations {{chat_history}}.

**Additional Info**
- Current Date : {{current_date}}
<<SYS>>
**Important Note** :
- Only use the context and the chat history to answer. If you don't know the answer, say so politely.
- Ask the user to check the College website for more information {{reference_links}} if you feel that the context is time sensitive.
- Only specify the time/date if the venue/date specified in the context is near to the current date, else ask them to check the website.
<</SYS>>
[/INST]
"""

@dataclass
class ConversationalConfig:
    """Configuration for conversational elements."""
    GREETING_LABELS: list = field(default_factory=lambda: [
        "hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon",
        "who are you ?", "what is your name ?", "introduce yourself", "tell me about yourself",
        "what can you do ?", "how can you help me ?", "what is your purpose ?", "what are you capable of ?",
        "what is your function ?", "what is your objective ?"
    ])

    SEND_OFF_LABELS: list = field(default_factory=lambda: [
        "thank", "thanks", "bye", "goodbye"
    ])
    
    GREETING_RESPONSES: list = field(default_factory=lambda: [
        "Hello! How can I assist you today regarding IIT Indore?",
        "Hi there! What would you like to know about the college?",
        "Greetings! I'm here to help with your IIT Indore questions.",
        "Hey! Ask me anything about IIT Indore.",
        "Hey! I'm here to provide information about IIT Indore. What's on your mind?",
        "Welcome! How can I guide you about the campus?"
    ])

    SEND_OFF_RESPONSES: list = field(default_factory=lambda: [
        "You're welcome! Feel free to ask if you have more questions.",
        "My pleasure!",
        "Glad to help! If you need more info, just let me know.",
        "Anytime! I'm here if you have more queries.",
        "It was nice chatting with you! Is there anything else about IIT Indore I can assist with?"
    ])
    INTENTS: list = field(default_factory=lambda: [
        "Admissions", "Academics", "Student Life", "Research", "Events"
    ])

# Create a single global configuration object
@dataclass
class AppConfig:
    """Overall application configuration."""
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    PATHWAY: PathwayConfig = field(default_factory=PathwayConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    BATCHING: BatchingConfig = field(default_factory=BatchingConfig)
    FLASK_APP: FlaskAppConfig = field(default_factory=FlaskAppConfig)
    PROMPTS: PromptsConfig = field(default_factory=PromptsConfig)
    CONVERSATION: ConversationalConfig = field(default_factory=ConversationalConfig)

# Instantiate the main configuration object
# This is the single point of access for all settings
config = AppConfig()

# from config import config
# model_choice = config.MODEL.MODEL_CHOICE
# pathway_host = config.PATHWAY.HOST
# log_file = config.DATA.INFO_LOGGING
# flask_secret = config.FLASK_APP.SECRET_KEY
# system_prompt = config.PROMPTS.SYSTEM_PROMPT
# intents = config.CONVERSATION.INTENTS
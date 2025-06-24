#file to define the constand, like fall back msgs, # default values, etc.
import os

# Option 1: Hardcoded path
DATA_PATH = "/home/jatin-sharma/Desktop/IITSoC/AIML-018-IITI-SoC/data"

# Option 2: Loaded from .env or env vars
DATA_PATH = os.getenv("DATA_PATH", "/default/path")

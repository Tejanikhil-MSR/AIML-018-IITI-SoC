import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "/home/jatin-sharma/Desktop/IITSoC/AIML-018-IITI-SoC/data")

DEVICE = os.getenv("DEVICE", "cpu")

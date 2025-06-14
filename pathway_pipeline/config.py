import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data")
DEVICE = os.getenv("DEVICE", "cuda:0")

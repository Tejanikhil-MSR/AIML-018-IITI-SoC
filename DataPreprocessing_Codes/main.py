import sys
sys.path.append("../CustomPathwayPipeline/")

from PDFSummarizer import PDFSummarizer
from config import UPDATED_DATA_DIR, MODEL_CHOICE
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model=MODEL_CHOICE, temperature=0.3)
summarizer = PDFSummarizer(model=model, output_dir=UPDATED_DATA_DIR)

summarizer.summarize_pdf("../Documentations/PG fee details_AY 2025-26.pdf")
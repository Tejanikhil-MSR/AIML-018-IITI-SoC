from PDFReader import PDFSummarizer
import pytesseract

ROOT_DIR = r"/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data/RAW_PDF_DOCS"
SAVE_DIR = r"../data/Summarized_PDFs"
MODEL_ENDPOINT = "http://0.0.0.0:11434/api/generate"
model = "llama3"
summarizer = PDFSummarizer(model_endpoint=MODEL_ENDPOINT, model=model, dataset_root_dir=ROOT_DIR)

summarizer.summarize_all_pdfs_and_save(SAVE_DIR)
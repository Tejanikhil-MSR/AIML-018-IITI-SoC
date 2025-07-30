import pathway as pw
import logging
from CustomPathwayPipeline.data_preprocessing import extract_keywords_from_string
from CustomPathwayPipeline.config import config
import asyncio
import time
logging.basicConfig(filename=config.DATA.DEBUGGER_LOGGING, filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_type(file_path) -> str:
    clean_file_path = str(file_path).strip('\"')
    if clean_file_path.endswith('.txt'):
        return "text"
    elif clean_file_path.endswith('.pdf'):
        return "pdf"
    else:
        logging.warning(f"[WARNING] Unsupported file type for path: {file_path}")
        return "unknown"

async def async_summarize_pdf(metadata, summarizer_service)->str:
    pdf_path = str(metadata).strip('\"')
    try:
        print(pdf_path)
        pdf_summary = await summarizer_service.summarize_pdf(str(pdf_path), save_as=False)
        logging.info(f"[INFO] Successfully summarized PDF: {pdf_summary}")
        return str(pdf_summary)
    
    except Exception as e:
        logging.error(f"[ERROR] Error summarizing PDF {pdf_path}: {e}")
        return f"Error summarizing PDF: {str(e)}"

def pre_process_text(text: str) -> str:
    import re
    try:
        clean_str = re.sub(r'[\r\n\t\\]+', ' ', text)
        clean_str = re.sub(r'[^\x20-\x7E.,!?()@&%#:/\-+_=\[\]{}\"\' ]+', '', clean_str) # Remove non-ASCII characters except common punctuation
        clean_str = re.sub(r'\s+', ' ', clean_str).strip()
        return clean_str
    
    except Exception as e:
        logging.error(f"[ERROR] Preprocessing error: {e}")
        return "Preprocessing error"

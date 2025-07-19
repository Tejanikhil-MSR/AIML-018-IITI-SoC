import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
import sys
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
sys.path.append("../")

from CustomPathwayPipeline.config import config

from CustomPathwayPipeline.data_preprocessing import PDFSummarizer, extract_keywords_from_string

import logging

logging.basicConfig(filename=config.DATA.DEBUGGER_LOGGING, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== initialize the required modules === 
model = ChatOllama(model=config.MODEL.MODEL_CHOICE, temperature=0.3)
summarizer_service = PDFSummarizer(model=model, summarize_prompt_template=config.PROMPTS.SUMMARIZATION_PROMPT, output_dir="PDFs_SUMMARIZED")

# === 1. Load data from filesystem in streaming mode (with metadata) ===
data_table = pw.io.fs.read(
    config.DATA.ROOT_DATA_DIR,
    format="binary",
    mode="streaming",
    with_metadata=True,
)

@pw.udf
def detect_type(file_path: str) -> str:
    import os
    clean_file_path = str(file_path).strip('\"')
    if clean_file_path.endswith('.txt'):
        return "text"
    elif clean_file_path.endswith('.pdf'):
        return "pdf"
    else:
        logging.warning(f"[WARNING] Unsupported file type for path: {file_path}")
        return "unknown"

@pw.udf(executor=pw.udfs.async_executor())
async def async_summarize_pdf(metadata:any)->str:
    pdf_path = str(metadata).strip('\"')
    try:
        print(pdf_path)
        pdf_summary = await summarizer_service.summarize_pdf(str(pdf_path), save_as=False)
        logging.info(f"[INFO] Successfully summarized PDF: {pdf_summary}")
        return pdf_summary
    
    except Exception as e:
        logging.error(f"[ERROR] Error summarizing PDF {pdf_path}: {e}")
        return f"Error summarizing PDF: {str(e)}"

typed_data = data_table.select(content=data_table["data"], path=data_table._metadata["path"], file_type=detect_type(pw.this._metadata["path"]), _metadata=data_table._metadata)

text_data = typed_data.filter(typed_data.file_type == "text")
pdf_data = typed_data.filter(typed_data.file_type == "pdf")

pdf_parsed = pdf_data.select(data=async_summarize_pdf(pw.this._metadata["path"]), _metadata=pw.this._metadata)
text_parsed = text_data.select(data=pw.apply(lambda b: b.decode("utf-8"), pw.this.content), _metadata=pw.this._metadata)

data = text_parsed.concat_reindex(pdf_parsed)

# After this the data_with_custom_metadata has (data: str, file_name: str, _metadata: pw.json("path":, "created_at":, "modified_at":, "seen_at": ,"size": ))
@pw.udf
def augment_metadata(text, metadata: pw.Json) -> dict:
    metadata = metadata.as_dict()
    metadata["filename"] = metadata["path"].split("/")[-1]
    metadata["keywords"] = "<keyword>".join(extract_keywords_from_string(text, 10))
    del metadata["path"]
    return metadata

data = data.with_columns(_metadata=augment_metadata(pw.this._metadata))

embeddings = HuggingFaceEmbeddings(model_name=config.PATHWAY.EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]  # Prioritize paragraph/sentence boundaries
)

server = VectorStoreServer.from_langchain_components(
    data,
    embedder=embeddings,
    splitter=splitter,
)

server.run_server(
    host=config.PATHWAY.HOST,
    port=config.PATHWAY.PORT,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem(config.PATHWAY.CACHE_DIR)
)

print(f"Pathway Vector Store Server running on {config.PATHWAY.HOST}:{config.PATHWAY.PORT}")
print(f"Monitoring directory: {config.DATA.ROOT_DATA_DIR}")
print(f"Embedding model: {config.PATHWAY.EMBEDDING_MODEL}")
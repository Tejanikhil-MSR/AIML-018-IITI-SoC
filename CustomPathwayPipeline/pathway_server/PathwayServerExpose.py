import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama

from config import PATHWAY_HOST, PATHWAY_PORT, EMBEDDING_MODEL, MODEL_CHOICE, CACHE_DIR, DEBUGGER_LOGGING, ROOT_DATA_DIR

from CustomPathwayPipeline.data_preprocessing import PDFSummarizer, extract_keywords_from_string

import logging

logging.basicConfig(filename=DEBUGGER_LOGGING, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== initialize the required modules === 
model = ChatOllama(model=MODEL_CHOICE, temperature=0.3)
summarizer_service = PDFSummarizer(model=model, output_dir="PDFs_SUMMARIZED")

# === 1. Load data from filesystem in streaming mode (with metadata) ===
data_table = pw.io.fs.read(
    ROOT_DATA_DIR,
    format="binary",
    mode="streaming",
    with_metadata=True,
)

@pw.udf
def detect_type(file_path: str) -> str:
    file_path= str(file_path)
    if file_path.endswith('.txt'):
        return "text"
    elif file_path.endswith('.pdf'):
        return "pdf"
    else:
        return "unknown"

@pw.udf
async def async_summarize_pdf(metadata:str)->str:
    try:
        pdf_summary = await summarizer_service.summarize_pdf(metadata, save_as=False)
        return pdf_summary
    
    except Exception as e:
        print(f"[ERROR] Error summarizing PDF {metadata}: {e}")
        return f"Error summarizing PDF: {str(e)}"

typed_data = data_table.select(content=data_table["data"], path=data_table._metadata["path"], file_type=detect_type(pw.this._metadata["path"]), _metadata=data_table._metadata)

text_data = typed_data.filter(typed_data.file_type == "text")
pdf_data = typed_data.filter(typed_data.file_type == "pdf")

pdf_parsed = pdf_data.select(data=pw.apply_async(async_summarize_pdf, pw.this._metadata["path"]), _metadata=pw.this._metadata)
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

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

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
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
    with_cache=True,
    cache_backend=pw.persistence.Backend.filesystem(CACHE_DIR)
)

print(f"Pathway Vector Store Server running on {PATHWAY_HOST}:{PATHWAY_PORT}")
print(f"Monitoring directory: {ROOT_DATA_DIR}")
print(f"Embedding model: {EMBEDDING_MODEL}")
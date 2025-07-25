import sys
sys.path.append("../")

import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from CustomPathwayPipeline.config import config
from CustomPathwayPipeline.data_preprocessing import PDFSummarizer, extract_keywords_from_string

import asyncio

import logging
logging.basicConfig(filename=config.DATA.DEBUGGER_LOGGING, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from helpers import detect_type, pre_process_text, async_summarize_pdf

# ==== initialize the required modules === 
model = ChatOllama(model=config.MODEL.MODEL_CHOICE, temperature=0.3)
summarizer_service = PDFSummarizer(model=model, summarize_prompt_template=config.PROMPTS.SUMMARIZATION_PROMPT, output_dir="PDFs_SUMMARIZED")

def server_init():

    @pw.udf(executor=pw.udfs.async_executor())
    async def parse_content(content: bytes, path: str) -> bytes:
        if detect_type(path)=="text":
            decoded_content = content.decode("utf8")
            processed_content = await asyncio.to_thread(pre_process_text, decoded_content)
            return processed_content.encode(encoding="utf-8", errors="strict")
        
        elif detect_type(path)=="pdf":
            summarized_content = await async_summarize_pdf(path, summarizer_service)
            return summarized_content.encode(encoding="utf-8", errors="strict")
        
        else:
            logging.warning(f"[WARNING] Unsupported file type for path: {path}")
            return b"Unsupported file type"

    @pw.udf
    def augment_metadata(text, metadata) -> dict:
        metadata = metadata.as_dict()
        metadata["filename"] = str(metadata["path"]).split("/")[-1]
        metadata["keywords"] = ", ".join(extract_keywords_from_string(text.decode('utf-8', 'strict'), 10))
        del metadata["path"]
        return metadata

    # === 1 : Load data from filesystem in streaming mode (with metadata) ===
    data_table = pw.io.fs.read(
        config.DATA.ROOT_DATA_DIR,
        format="binary",
        mode="streaming",
        with_metadata=True, 
        autocommit_duration_ms=1500, 
    )
    # Expected Schema: (data_table)
    # |--- data          : bytes
    # |--- _metadata     : pw.Json(path: str, created_at: int, modified_at: int, seen_at: int, size: int)
    
    # === 2 : parse the data based on the file_types ===
    data = data_table.select(data=parse_content(data_table.data, data_table._metadata["path"]), _metadata=data_table._metadata)
    # Expected Schema: (data)
    # |--- data          : bytes
    # |--- _metadata     : pw.Json(path: str, created_at: int, modified_at: int, seen_at: int, size: int)
    
    # === 3 : augment metadata with additional information ===
    data = data.with_columns(_metadata=augment_metadata(pw.this.data, pw.this._metadata))
    # Expected Schema:
    # |--- data          : bytes
    # |--- _metadata     : pw.Json(filename: str, keywords: str, created_at: int, modified_at: int, seen_at: int, size: int)
    # Note: The original 'path' from _metadata is replaced by 'filename' and 'keywords' are added.
    
    # pw.debug.compute_and_print(data, include_id=False)

    # === 4 : Initialize the embedder from hugging face ===
    embeddings = HuggingFaceEmbeddings(model_name=config.PATHWAY.EMBEDDING_MODEL,  model_kwargs = {'device': 'cpu'})

    # === 5 : Initialize a chunker to split the documents ===
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]  # Prioritize paragraph/sentence boundaries
    )

    # === 6 : Create the Vector Store Server ===
    server = VectorStoreServer.from_langchain_components(
        data,
        embedder=embeddings,
        splitter=splitter,
    )

    return server

if __name__=="__main__":

    server = server_init()

    # === 7 : start the server with caching enabled ===
    server.run_server(
        host=config.PATHWAY.HOST,
        port=config.PATHWAY.PORT,
        with_cache=True,
        cache_backend=pw.persistence.Backend.filesystem(config.PATHWAY.CACHE_DIR)
    )


    print(f"Pathway Vector Store Server running on {config.PATHWAY.HOST}:{config.PATHWAY.PORT}")
    print(f"Monitoring directory: {config.DATA.ROOT_DATA_DIR}")
    print(f"Embedding model: {config.PATHWAY.EMBEDDING_MODEL}")
from dotenv import load_dotenv
import os

import pathway as pw
from data_ingestion import read_documents
from embedding import get_embedder
from parsing import get_parser
from splitting import get_splitter
from retrieval import get_retriever_factory, get_document_store
from serving import setup_webserver, setup_rest_connector, QuerySchema
from prompts import build_prompts_udf
from pathway.xpacks.llm import llms

load_dotenv()

DATA_PATH = os.getenv("pathway_monitoring_folder")
use_gpu = os.getenv("USE_GPU")
if(use_gpu == "true"):
    DEVICE = "gpu"
else:
    DEVICE = "cpu"


# Ingest documents
documents = read_documents(DATA_PATH)

# Set up pipeline components
embedder = get_embedder(device=DEVICE)
retriever_factory = get_retriever_factory(embedder)
parser = get_parser()
splitter = get_splitter()
document_store = get_document_store(documents, retriever_factory, parser, splitter)

# Set up webserver and REST connector
webserver = setup_webserver()
queries, writer = setup_rest_connector(webserver)

queries = queries.select(query=pw.this.messages, k=1, metadata_filter=None, filepath_globpattern=None)

retrieved_documents = document_store.retrieve_query(queries)
retrieved_documents = retrieved_documents.select(docs=pw.this.result)
queries_context = queries + retrieved_documents

prompts_with_context = queries_context.with_columns(
    prompt=build_prompts_udf(pw.this.docs, pw.this.query)
)

model = llms.HFPipelineChat(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=DEVICE
)

responses = prompts_with_context.with_columns(
    result=model(llms.prompt_chat_single_qa(pw.this.prompt))
)

writer(responses)
pw.run()

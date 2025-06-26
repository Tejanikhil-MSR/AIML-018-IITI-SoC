from dotenv import load_dotenv
import os
import time
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

DATA_PATH = "/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data_clean_50"
# use_gpu = os.getenv("USE_GPU")
# if(use_gpu == "true"):
DEVICE = "cuda:0"
# else:
    # DEVICE = "cpu"


# Ingest documents
print("Starting document loading...")
start_load_time = time.time()
documents = read_documents(DATA_PATH)
end_load_time = time.time()
print(f"Document loading completed in: {end_load_time - start_load_time:.2f} seconds")



print("Setting up document processing components...")
start_setup_time = time.time()

# Set up pipeline components
embedder = get_embedder(device=DEVICE)
retriever_factory = get_retriever_factory(embedder)
parser = get_parser()
splitter = get_splitter()

end_setup_time = time.time()
print(f"Component setup completed in: {end_setup_time - start_setup_time:.2f} seconds")

# Create document store with timing
print("Creating document store and processing documents...")
start_docstore_time = time.time()

document_store = get_document_store(documents, retriever_factory, parser, splitter)


end_docstore_time = time.time()
print(f"Document store creation and document processing completed in: {end_docstore_time - start_docstore_time:.2f} seconds")
# Set up webserver and REST connector
webserver = setup_webserver()
queries, writer = setup_rest_connector(webserver)

# from the queries table select the 
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
pw.run( #debug=True,                           # Enable debug output
    #monitoring_level=pw.MonitoringLevel.NONE,  # Disable monitoring dashboard
     default_logging=True,                 # Keep default logging (optional)
    terminate_on_error=False  )

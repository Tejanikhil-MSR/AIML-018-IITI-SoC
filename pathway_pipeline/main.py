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

DATA_PATH = "/home/jatin-sharma/Desktop/IITSoC/AIML-018-IITI-SoC/data"
use_gpu = False
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

# from the queries table select the 
# queries = queries.select(query=pw.this.messages, k=1, metadata_filter=None, filepath_globpattern=None)


conversation_memory = queries.select(
    session_id=pw.this.session_id,
    message=pw.this.messages
)

grouped_memory = (
    conversation_memory
    .groupby(pw.this.session_id)
    .reduce(
        session_id=pw.this.session_id,
        history=pw.reducers.join_strings(pw.this.message, separator="\n")
    )
)

queries_with_history = queries.join(
    grouped_memory, on=pw.this.session_id == grouped_memory.session_id
)


# Now: queries_with_history has: session_id, messages, history
queries_contextual = queries_with_history.select(
    query=pw.this.messages,
    history=pw.this.history,
    k=1,
    metadata_filter=None,
    filepath_globpattern=None
)

# queries  = queries_contextual +queries
retrieved_documents = document_store.retrieve_query(queries_contextual)
retrieved_documents = retrieved_documents.select(docs=pw.this.result)
queries_contextual = queries_contextual + retrieved_documents


# prompts_with_context = queries_context.with_columns(
#     prompt=build_prompts_udf(pw.this.docs, pw.this.query)
# )
prompts_with_context = queries_contextual.with_columns(
    prompt=build_prompts_udf(pw.this.docs, pw.this.query, pw.this.history)
)

model = llms.HFPipelineChat(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=DEVICE
)

# responses = prompts_with_context.with_columns(
#     result=model(llms.prompt_chat_single_qa(pw.this.prompt))
# )

responses = prompts_with_context.with_columns(
    result=model(llms.prompt_chat_single_qa(pw.this.prompt))
)

# Add response back to conversation memory
new_conversation_entries = responses.select(
    session_id=pw.this.session_id,
    message=pw.this.query + "\n" + pw.this.result  # both question and response
)

# Combine old and new memory
conversation_memory += new_conversation_entries  # pseudo-code; use Pathway ops

writer(responses)
pw.run()
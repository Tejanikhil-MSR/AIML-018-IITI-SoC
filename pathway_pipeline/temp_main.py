from dotenv import load_dotenv
import os
from json import dumps
import pathway as pw
from data_ingestion import read_documents
from embedding import get_embedder
from parsing import get_parser
from splitting import get_splitter
from retrieval import get_retriever_factory, get_document_store
from serving import setup_webserver, setup_rest_connector, QuerySchema
from prompts import build_prompts_udf # Keep if you use it elsewhere, though build_prompt_with_history replaces it here
from pathway.xpacks.llm import llms
from collections import deque
import json

load_dotenv()

DATA_PATH = "/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data"
DEVICE = "cuda:0"

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
queries = queries.select(query=pw.this.messages, k=1, metadata_filter=None, filepath_globpattern=None)

# Initialize a table to store query history
# This table will accumulate past queries and their responses
query_history = pw.Table.empty(
    query=str,
    response=str,
    doc_context=str, # Store the context that was used for the response
)

# Retrieve documents for the current query
retrieved_documents = document_store.retrieve_query(queries)
retrieved_documents = retrieved_documents.select(
    docs=pw.this.result,
    query=queries.query, # Pass query along for joining later
)

# Join queries with their retrieved documents
# Use a left join to ensure all queries are processed, even if no documents are retrieved

queries_with_docs = queries.join_left(retrieved_documents, pw.this.query == pw.right.query)
print(queries_with_docs)
# Ensure 'docs' column exists for all queries, even if no join occurred
queries_with_docs = queries_with_docs.select(docs=pw.this.docs)
queries_with_docs = queries_with_docs.with_columns(
    docs=pw.if_else(pw.this.docs.is_not_none(), pw.this.docs, "")
)

# UDF to build the prompt with history
@pw.udf
def build_prompt_with_history(current_query: str, retrieved_docs: str, history_data: list) -> str:
    # `history_data` will be a list of dicts: [{"query": ..., "response": ..., "doc_context": ...}]
    # We'll take the last few turns for context
    context_parts = []
    # Take up to the last 3 turns of conversation
    for turn in history_data[-3:]:
        context_parts.append(f"User: {turn['query']}")
        context_parts.append(f"Assistant: {turn['response']}")

    conversation_context = "\n".join(context_parts)

    if conversation_context:
        conversation_context = f"Conversation History:\n{conversation_context}\n\n"

    # Combine context and current query
    prompt = (
        f"{conversation_context}"
        f"Retrieved Documents:\n{retrieved_docs}\n\n"
        f"User: {current_query}\n"
        f"Assistant:"
    )
    return prompt

prompts_with_context = queries_with_docs.with_columns(
    prompt=pw.apply_fn(
        build_prompt_with_history,
        pw.this.query,
        pw.this.docs, # Pass the retrieved docs to the UDF
        # IMPORTANT: The history_data argument here cannot directly be `query_history` table
        # Instead, we need to pass a representation of the history that is compatible
        # with Pathway's UDF execution model.
        # This is where the 'universe' error comes from. `query_history` is a separate table.
        # To get around this, for a simple global history, we can aggregate.
        # However, for a *streaming* context, passing the entire history to every UDF call
        # is inefficient and often incorrect.

        # A more robust solution for persistent, per-session history involves
        # maintaining session state.
        # For a simpler, global "last N turns" without explicit session management,
        # you often need to aggregate the history into a single row/value that can be broadcast.

        # Let's try a different approach to include history:
        # Instead of dynamically joining, pass the current accumulated history
        # as a *scalar value* derived from `query_history`.

        # This requires `query_history` to be aggregated into a single record
        # if we want to add it to every incoming query.

        # Let's modify the UDF to expect `history_data` as a list of dictionaries
        # (e.g., from a JSON string, which is serializable).

        # Option 1: Global History Aggregation (simplistic, for demo)
        # This is not scalable for many concurrent users, but illustrates the concept.
        # Aggregate the `query_history` into a single, summary string for all queries.
    )
)

# A more appropriate way to handle history in Pathway is by joining on a common key (like a session ID).
# Since you don't have one, let's simulate the common case where history is passed *along with* the query
# (e.g., from a client that maintains conversation state).
# Or, if we assume a single user and want the "last N global turns," we need a pattern to make `query_history`
# accessible.

# **Revised approach for history management in Pathway for a streaming RAG chatbot:**
# 1. The `query_history` table accumulates all Q&A pairs.
# 2. When a *new* query comes in, we want to look at the *current state* of `query_history`.
# 3. This often means doing a `join` or using `pw.reduce` to get the necessary historical context.

# Let's use a `pw.reduce` to get the last N turns of the history as a single value.
# This is a bit of a hack for "global history" but demonstrates how to get data across tables.

# Aggregate the `query_history` into a single string that can be joined.
# This will always reflect the *entire* history (or a truncated version of it).
# This is generally not ideal for multi-user systems without a session key.

# Let's re-think the `build_prompt_with_history` UDF.
# It should take `current_query` and `retrieved_docs` directly.
# The `history` part needs to be handled outside, by joining a *transformed* history table.

# Corrected approach to bring history to the current query:
# We need to make `query_history`'s data available for each *new* incoming query.
# This is effectively a "broadcast join" or "cross join" if there's no common key.

# Let's assume we want to append the *entire* (or last N) conversation history to *each* new query.

# Step 1: Create a view of `query_history` that represents the "last N" turns.
# This is tricky with Pathway's streaming nature. `pw.window` is typically for time-based windows.

# For a "last N items" in a streaming table, you might need a custom UDF that maintains state or
# convert the history to a single row to join.

# Let's define the `build_prompt_with_history` UDF expecting `history` as a string or list of dicts.
# The challenge is how to get this `history` from `query_history` to `queries_with_docs`.

# The `ValueError: You cannot use <table1>.history in this context. Its universe is different...`
# means that `recent_history.history` is a column belonging to a table that Pathway doesn't
# consider "joinable" directly with `queries_context` in that `with_columns` call.

# To fix this, you need to ensure the data you're joining has a compatible universe.
# If you want to take a snapshot of `query_history` and join it with *every* incoming query,
# you typically need to reduce `query_history` to a single row containing the aggregated history,
# and then cross-join it (or left_join on a constant, if supported).

# Let's create a "global history" table that is a single row with the concatenated history.

# Global history table (this will update as query_history updates)
# This is a bit of a hack to get a "global" history into every query.
# In a real multi-user scenario, you'd have a `user_id` for joining.
global_history = query_history.group_by().reduce(
    history=pw.reducers.concat(pw.this.query + ": " + pw.this.response + "\n", separator="")
).select(history=pw.this.history)


# Now, join the incoming queries with this aggregated global history.
# Since global_history has only one row after group_by().reduce(),
# a join on a constant or simply adding it will effectively "broadcast" it.

queries_with_global_history = queries_with_docs.with_columns(
    history=global_history.history, # This will try to join based on universe,
                                     # which is still the problem.
)

# To truly add it, you might need a cross-join equivalent or a different UDF structure.

# **Alternative for history: Pass a dynamic snapshot via a UDF that *closes over* `query_history` (less Pathway-idiomatic)**
# This is generally discouraged as UDFs are meant to be pure functions of their inputs.

# **The most Pathway-idiomatic way for history (assuming a session key):**
# If `queries` had a `session_id`, you would join `queries` with `query_history` on `session_id`.
# Since it doesn't, we need to decide if history is global or implied per "session".

# If we *must* use `build_prompt_with_history(pw.this.history, pw.this.query)`
# and `history` needs to come from `recent_history`, then `queries_context` and `recent_history`
# *must* have compatible keys, or you need to do an explicit join that produces compatible keys.

# Let's assume you want the *entire current* `query_history` for each new incoming query.
# This is a cross-join. Pathway's `join` needs a condition. A common pattern for cross-join
# is to create a dummy key.

# Create a dummy key for cross-joining
queries_with_dummy_key = queries_with_docs.with_columns(dummy_key=1)
global_history_with_dummy_key = global_history.with_columns(dummy_key=1)

# Perform a join based on the dummy key
# This effectively creates a cross-join: each query gets the aggregated history.
queries_context_with_history = queries_with_dummy_key.join(
    global_history_with_dummy_key,
    pw.this.dummy_key == pw.this.dummy_key
).select(
    query=pw.this.query,
    docs=pw.this.docs,
    history=pw.this.history, # This is now the column from the joined global_history
)

# Now, `queries_context_with_history` has `query`, `docs`, and `history`.
prompts_with_context = queries_context_with_history.with_columns(
    prompt=build_prompt_with_history(pw.this.query, pw.this.docs, pw.this.history)
)

# Make sure the UDF `build_prompt_with_history` correctly handles `history`
# when it's passed as a single string (from `global_history`).
# The UDF should be:
@pw.udf
def build_prompt_with_history(current_query: str, retrieved_docs: str, history_str: str) -> str:
    # `history_str` will be the concatenated string of previous Q&A turns
    conversation_context = ""
    if history_str:
        conversation_context = f"Conversation History:\n{history_str}\n\n"

    # Combine context and current query
    prompt = (
        f"{conversation_context}"
        f"Retrieved Documents:\n{retrieved_docs}\n\n"
        f"User: {current_query}\n"
        f"Assistant:"
    )
    return prompt

# Re-apply the UDF call now that the history is structured as a single string
prompts_with_context = queries_context_with_history.with_columns(
    prompt=build_prompt_with_history(pw.this.query, pw.this.docs, pw.this.history)
)


model = llms.HFPipelineChat(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=DEVICE
)

responses = prompts_with_context.with_columns(
    result=model(llms.prompt_chat_single_qa(pw.this.prompt))
)

# Update the query_history table with the new query and its response, and the context used.
query_history += responses.select(
    query=pw.this.query,
    response=pw.this.result,
    doc_context=pw.this.docs, # Store the documents used for this response
)

writer(responses)

pw.run( debug=True,
    monitoring_level=pw.MonitoringLevel.NONE,
    default_logging=True,
    terminate_on_error=False
)
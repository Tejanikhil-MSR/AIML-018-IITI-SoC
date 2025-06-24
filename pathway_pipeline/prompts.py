import pathway as pw

def get_context(documents):
    content_list = []
    for doc in documents:
        content_list.append(str(doc["text"]))
    return " ".join(content_list)


@pw.udf
def build_prompts_udf(documents, query,history) -> str:
    context = get_context(documents)
    history = history if history else ""
    history_context = get_context(history)

    prompt = (
        f"Given the following documents : \n {context}\n and history context as : {history_context}:\nanswer this query: {query}"
    )
    return prompt

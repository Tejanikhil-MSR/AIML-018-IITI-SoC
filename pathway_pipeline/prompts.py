import pathway as pw

def get_context(documents):
    content_list = []
    for doc in documents:
        content_list.append(str(doc["text"]))
    return " ".join(content_list)

@pw.udf
def build_prompts_udf(documents, query) -> str:
    context = get_context(documents)
    prompt = (
        f"Given the following documents : \n {context} \nanswer this query: {query}"
    )
    return prompt

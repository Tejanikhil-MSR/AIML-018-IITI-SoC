from pathway.xpacks.llm.splitters import TokenCountSplitter

def get_splitter():
    return TokenCountSplitter(
        min_tokens=100, max_tokens=500, encoding_name="cl100k_base"
    )

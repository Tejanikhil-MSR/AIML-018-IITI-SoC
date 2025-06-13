from pathway.xpacks.llm.parsers import UnstructuredParser

def get_parser():
    return UnstructuredParser(
        chunking_mode="by_title",
        chunking_kwargs={
            "max_characters": 3000,
            "new_after_n_chars": 2000,
        },
    )

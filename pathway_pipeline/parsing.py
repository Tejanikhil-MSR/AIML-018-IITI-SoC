from pathway.xpacks.llm.parsers import UnstructuredParser, Utf8Parser

def get_parser():
    return Utf8Parser(
        # chunking_mode="paged",
        # chunking_kwargs={
        #     "max_characters": 3000,
        #     "new_after_n_chars": 2000,
        # },
    )


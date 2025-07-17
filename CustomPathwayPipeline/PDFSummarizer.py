import os
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PDFSummarizer:
    
    def __init__(self, model, output_dir: str):
        """
        Initialize the PDF summarizer.

        Args:
            model: A language model instance (e.g., ChatOllama with a given model name).
            output_dir: Directory to write the output summaries to.
        """
        self.model = model
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Prepare summarization prompt
        summarize_prompt_template = """
        You are an AI assistant tasked with summarizing content.
        Provide a concise and informative summary of the following text or table content.
        Focus on the main points and key information.

        Content:
        {element}

        Summary:
        """
        
        summarize_prompt = ChatPromptTemplate.from_template(summarize_prompt_template)
        self.summarize_chain = summarize_prompt | self.model | StrOutputParser()
    
    def summarize_pdf(self, pdf_path: str, save_as=True):
        
        if not os.path.exists(pdf_path):
            print(f"[ERROR] PDF not found: {pdf_path}")
            return

        pdf_filename = os.path.basename(pdf_path)
        output_file_path = os.path.join(self.output_dir, f"{os.path.splitext(pdf_filename)[0]}.txt")

        try:
            chunks = partition_pdf(filename=pdf_path, infer_table_structure=True, strategy="hi_res", extract_image_block_to_payload=False,
                                   chunking_strategy="by_title", max_characters=4000, combine_text_under_n_chars=1000, new_after_n_chars=3000,)
            
        except Exception as e:
            print(f"[ERROR] Failed to extract PDF: {e}")
            return

        texts, tables = [], []
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            elif any(t in str(type(chunk)) for t in ["CompositeElement", "NarrativeText", "Title"]):
                texts.append(chunk)

        all_summaries = []

        for chunk in texts:
            try:
                summary = self.summarize_chain.invoke({"element": chunk.text})
                all_summaries.append(summary)
            except Exception as e:
                all_summaries.append(chunk.text[:200])

        for chunk in tables:
            table_content = getattr(chunk.metadata, "text_as_html", chunk.text)
            try:
                summary = self.summarize_chain.invoke({"element": table_content})
                all_summaries.append(summary)
            except Exception as e:
                all_summaries.append(table_content[:200])

        if all_summaries:
            combined_summary_input = "\n\n".join(all_summaries)
            try:
                final_summary = self.summarize_chain.invoke({"element": combined_summary_input})
            except Exception as e:
                final_summary = combined_summary_input
        else:
            final_summary = "No content found in PDF."

        if save_as:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"--- Summary for {pdf_filename} ---\n\n")
                f.write(final_summary)
                f.write("\n")
            print(f"✅ Summary written to: {output_file_path}")

        return final_summary

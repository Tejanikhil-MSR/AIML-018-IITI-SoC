import layoutparser as lp
import cv2
import requests
import re
import os
from pdf2image import convert_from_path
import numpy as np
import nltk
from nltk.corpus import words
nltk.download('words')
from pathlib import Path

class PDFSummarizer:

    def __init__(self, dataset_root_dir, model, model_endpoint, with_llm:bool = True):
        self.dataset_root_dir = dataset_root_dir
        self.model = model
        self.model_endpoint = model_endpoint
        self.with_llm = with_llm
    
    @staticmethod
    def doc_to_imgs(filename: str):
        # Convert PDF to images
        images = convert_from_path(filename)
        image_list = []
        
        for image in images:
            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_list.append(image_cv)

        return image_list
    
    @staticmethod
    def filter_text(text: str) -> str:
        """Function that filters out the meaningless english text from the text."""
        ALLOWED_SYMBOLS = r".,:;'\-_/\\()\""
        english_vocab = set(words.words())
        tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text)
        valid_tokens = [t for t in tokens if t.lower() in english_vocab or t.isdigit() or any(c in t for c in ALLOWED_SYMBOLS)]
        return ' '.join(valid_tokens)

    @staticmethod
    def extract_text_from_image(image: np.ndarray) -> str:
        
        ocr_agent = lp.TesseractAgent(languages='eng')

        # Use the OCR agent to extract text
        text = ocr_agent.detect(image)

        return text
    
    def _get_all_pdfs(self):
        root_path = Path(self.dataset_root_dir)
        return list(root_path.rglob("*.pdf"))

    @staticmethod
    def ask_ollama(prompt: str, model: str, endpoint: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(endpoint, json = payload)
        return response.json().get('response', '')

    @staticmethod
    def is_hindi_doc(filename: str) -> bool:
        """Function to check if the document is in Hindi using the filename."""
        filename = str(filename).lower()
        if re.search(r'\bhindi\b', filename):
            return True
        return False

    def _make_summarization_prompt(self, text: str) -> str:

        prompt =  f""" You are a text processing assistant tasked with transforming provided text into a coherent paragraph while preserving all key details.
        
        ## Instructions:  
        1. **Classify the document** into one of the following categories:  
            - Academics  
            - Club-Activities/Events/Conferences/Hackathons
            - Interviews/Faculty-Openings/PhD-Opportunities
            - Startups/Companies/Institutions/Collaborations  
            - Patents/Achievements/Contributions
            - Tenders/Procurements/Quotations
            - Rules&Regulations/Policies
            - Others

        2. **Create a coherent paragraph** from the text based on the classification, ensuring:  
            - Include all key details (e.g., dates, email addresses, eligibility criteria, application instructions, deadlines if specified).  
            - For **Club-Activities/Events/Conferences/Hackathons**: Include name, date, location, target audience, requirements, deadlines, any other details specified in the document.  
            - For **Patents/Achievements/Contributions**: Include all the details that is specific to the document and state clearly what is the topic/area in which they achieved.  
            - For **Rules&Regulations/Policies**: Include rule, purpose, and consequences/requirements for violations and any other details mentioned.  
            - For **Interviews/Faculty-Openings/PhD-Opportunities**: Include role, whom it is concerned to, location, requirements, skills, deadlines, any other details specified in the document.  
            - For **Tenders/Procurements/Quotations**: Include all relevant details from the text.  
            - For **Others**: Include all information from the text.  
            - Integrate keywords naturally to ensure clarity if the paragraph is split.  
            - Write as if directly informing the reader, avoiding phrases like “The document states” or “This is about.”  
            - Use only the provided text, without external knowledge.

        ## Inpput Format Explained: 
        - It is text extracted from a pdf document, which may contain various types of information such as academics, club activities, interviews, patents, rules and regulations, etc.
        - The text may contain multiple sections, broken sentences (abrupltly ending), might also contain spelling mistakes and may not be in a coherent format

        ## Output Format:
        - Return the result in this **exact format** (without additional commentary):
        ```plaintext
        <Classification>
        <Coherent paragraph on the second line>
        ```

        ## Important Notes:
        - Ensure that the classification is accurate and the paragraph is coherent and try to include as unique keywords as possible that are relavent to the class in the paragraph so that it will be easier to preserve the context even if i split the data into chunks.
        - write the summary as if you are directly informing the reader, avoiding phrases like "The document states" or "This is about."
        - Do not include any summary notes, disclaimers, or commentary like “this summary only covers
        - Do not prefix the output with any phrases like “Here is the output,” “Summary,” “Classification,” or anything similar.
        - Absolutely NO introductory statements or explanation of the result.
        - Output must begin directly with the classification, followed by the summary paragraph on the next line.
        - **Important**: Do not add any extra commentary, greetings, or closing statements after the output.
        - If not text is present in the input then return 
            ```
            Empty Document
            No text found in the document.
            ```

        ## Example Output:
        Interviews/Faculty-Openings/PhD-Opportunities
        The Department of Computer Science at XYZ University is inviting applications for faculty positions at the Assistant Professor level. Interested candidates must have a PhD in Computer Science or a related field, a strong research record, and teaching experience. Applications must be submitted by September 15, 2025, via email to recruitment@xyzuniversity.edu. Additionally, PhD opportunities are open for highly motivated candidates with a Master's degree and a valid GATE/NET qualification. Shortlisted applicants will be called for interviews in October 2025 at the university campus.

        ## Text to process:
        ```
        {text}
        ```

        """
        return prompt

    def _summarize_text(self, text: str) -> str:
        """Function used for summarizing the tables of a pdf document."""
        prompt = self._make_summarization_prompt(text)
        response = self.ask_ollama(prompt, self.model, self.model_endpoint)
        return response

    def extract_text_from_doc(self, pdf_path: str) -> str:
        images = self.doc_to_imgs(pdf_path)
        text = ""
        for image in images:
            extracted_text = self.extract_text_from_image(image)
            filtered_text = self.filter_text(extracted_text)
            text += filtered_text + "\n"

        return text.strip()

    def summarize_pdf(self, pdf_path: str) -> str:
        """Function used for summarizing the pdf document."""
        if self.is_hindi_doc(pdf_path):
            print("[ERROR] is a hindi document, skipping summarization.")
            return ""

        images_converted = self.doc_to_imgs(pdf_path)

        text = ""
        for image in images_converted:
            temp = self.extract_text_from_image(image)
            filtered_text = self.filter_text(temp)
            text += filtered_text + "\n"
        
        summary = self._summarize_text(text)

        return summary
    
    def summarize_all_pdfs(self):
        summary = []
        pdf_documents_list = self._get_all_pdfs()
        for pdf_path in pdf_documents_list:
            if self.is_hindi_doc(pdf_path):
                print("[ERROR] is a hindi document, skipping summarization.")

            path = os.path.join(self.dataset_root_dir, pdf_path)
            summary.append(self.summarize_pdf(path))

        return summary
    
    def get_all_files_in_directory(self, directory, format=".txt"):
        all_files = []
        filenames = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(format):
                    all_files.append(os.path.join(root, file))
                    filenames.append(file)
        return all_files, filenames

    def summarize_all_pdfs_and_save(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        [documents_list,filenames] =self.get_all_files_in_directory(self.dataset_root_dir, format=".pdf")
        counter = 0
        for pdf_path,file_name in zip(documents_list,filenames):
            if self.is_hindi_doc(pdf_path):
                print("[ERROR] is a hindi document, skipping summarization.")
                continue
            path = pdf_path

            summary = self.summarize_pdf(path)
            filename = f"{file_name}.txt"
            output_file = save_path / filename
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary)

            print(f"[✓] PDF_ID : {counter} Saved as: {output_file}")
            counter += 1
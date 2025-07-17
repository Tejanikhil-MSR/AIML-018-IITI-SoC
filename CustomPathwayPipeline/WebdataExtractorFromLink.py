import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging 
from ParseLogs import LogParser
from config import Text_DATA_DIR, PDF_DATA_DIR

class KnowledgeBaseUpdater:
    
    def __init__(self, text_files_directory, pdf_files_directory):
        self.text_files_directory = text_files_directory
        self.pdf_files_directory = pdf_files_directory 

    def _sanitizeFilename(self, url):
        return urlparse(url).netloc.replace('.', '_') + '_' + os.path.basename(url).replace('/', '_')

    def _extractTextAndPdfs(self, url):
        
        try:
            response = requests.get(url,verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')


            text = soup.get_text()
            domain = urlparse(url).netloc.replace('.', '_')
            os.makedirs(self.text_files_directory, exist_ok=True)
        
            with open(os.path.join(self.text_files_directory, f'{domain}__page_text.txt'), 'w', encoding='utf-8') as f:
                f.write(text.strip())

            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.lower().endswith('.pdf'):
                    full_url = urljoin(url, href)
                    pdf_links.append(full_url)

            print(f"Found {len(pdf_links)} PDF(s). Downloading...")
            
            os.makedirs(self.pdf_files_directory, exist_ok=True)
                
            for pdf_url in pdf_links:
                try:
                    pdf_resp = requests.get(pdf_url)
                    pdf_resp.raise_for_status()
                    filename = self._sanitizeFilename(pdf_url)
                    filepath = os.path.join(self.pdf_files_directory, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(pdf_resp.content)
                    print(f"Downloaded: {pdf_url}")
                except Exception as e:
                    print(f"Failed to download {pdf_url}: {e}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
        return True
            
    def update(self, urls):
        logging.info(f"Knowledge update started...... using logs from ./logs/info.log")
        for url in urls:
            updated = self._extractTextAndPdfs(url)
        
            if(updated==True):
                logging.info(f"Knowledge base updated successfully from {url}")
                
            else:
                logging.warning(f"Knowledge base updated unsuccessful from {url}")
        
webdata_updater = KnowledgeBaseUpdater(text_files_directory=Text_DATA_DIR, pdf_files_directory=PDF_DATA_DIR)
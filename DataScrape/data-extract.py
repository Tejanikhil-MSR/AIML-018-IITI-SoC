import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()

visit_lock = threading.Lock()
visit = set()
to_visit = queue.Queue()

disallowed_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg','.webp', '.ico', '.mp4', '.mp3', '.avi', '.mov', '.zip', '.rar')

def safe_path_from_url(pdf_url):
    parsed = urlparse(pdf_url)
    path = parsed.path.lstrip('/')
    return parsed.netloc.replace('.', '^^') + '>>' + path.replace('/', '??')

def is_valid_link(href):
    if not href.startswith(('http://', 'https://')):
        return False
    parsed = urlparse(href)
    domain = parsed.netloc
    if not (domain == 'iiti.ac.in' or domain.endswith('.iiti.ac.in')):
        return False
    clean_path = parsed.path.lower()
    return not any(clean_path.endswith(ext) for ext in disallowed_extensions)

def extract_and_save(url, output_dir):
    try:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Download PDFs
        domain = urlparse(url).netloc.replace('.', '^^')
        text_dir = os.path.join(output_dir, domain)
        os.makedirs(text_dir, exist_ok=True)

        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = urljoin(url, href)
                try:
                    pdf_resp = requests.get(full_url, verify=False)
                    pdf_resp.raise_for_status()
                    filename = safe_path_from_url(full_url)
                    filepath = os.path.join(text_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(pdf_resp.content)
                except Exception as e:
                    print(f"Failed to download {full_url}: {e}")

        # Parse and collect links
        soup = BeautifulSoup(response.content, 'lxml')
        new_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            full_link = urljoin(url, href)
            if is_valid_link(full_link):
                with visit_lock:
                    if full_link not in visit:
                        visit.add(full_link)
                        new_links.append(full_link)
                        to_visit.put(full_link)

        # Remove anchors and get clean text
        soup1 = BeautifulSoup(response.content, 'html.parser')
        for a_tag in soup1.find_all('a', href=True):
            a_tag.decompose()
        text = soup1.get_text(separator='\n', strip=True)

        # Save page text
        if text:
            with open(os.path.join(text_dir, 'page_text.txt'), 'w', encoding='utf-8') as f:
                f.write(text)

    except Exception as e:
        print(f"Error fetching {url}: {e}")

def worker(output_dir):
    while True:
        try:
            url = to_visit.get()
            extract_and_save(url, output_dir)
            to_visit.task_done()
        except queue.Empty:
            break

def main(start_url, output_dir):
    with visit_lock:
        visit.add(start_url)
    to_visit.put(start_url)

    with ThreadPoolExecutor(max_workers=16) as executor:
        for _ in range(16):
            executor.submit(worker, output_dir)

    to_visit.join()

if __name__ == "__main__":
    main("https://iiti.ac.in", "dataset/")

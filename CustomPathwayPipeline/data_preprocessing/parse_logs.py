import re
from pathlib import Path
from typing import List, Optional
from config import Logdir
import os

class LogParser:
    def __init__(self):
        # nothing to be instantiated
        pass
    
    @staticmethod
    def get_latest_created_folder(base_path) -> Optional[Path]:
        folders = [f for f in base_path.iterdir() if f.is_dir()]
        if not folders:
            return None
        folders.sort(key=lambda f: f.stat().st_ctime)
        return folders[-1]

    def _parse_log_file(self, log_file_path: Path) -> List[str]:
        retrieved_links = []

        with log_file_path.open('r') as file:
            for line in file:
                if "Files retrieved" in line:
                    match = re.search(r'Files retrieved\s*:\s*\[(.*?)\]', line)
                    if match:
                        links_raw = match.group(1)
                        links = [link.strip() for link in links_raw.split(',')]
                        retrieved_links.extend(links)

        return list(set(retrieved_links))  # Remove duplicates
    
    def _convert_into_link(filename: str) -> str:
        regenerated_link = filename
        return regenerated_link

    def parse(self) -> List[str]:
        
        links = []

        log_file_path = os.path.join(Logdir, "info.log")
        if not log_file_path.exists():
            raise FileNotFoundError(f"No log file found at: {log_file_path}")

        retrieved_filepaths = self._parse_log_file(log_file_path)
        
        for file_path in retrieved_filepaths:
            links.append(self._convert_into_link(file_path))
        
        return list(set(links))

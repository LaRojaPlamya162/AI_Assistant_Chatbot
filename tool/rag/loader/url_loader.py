from typing import List
from langchain_core.documents import Document
import requests
from .base import BaseLoader
from component.tool import load_pdf_from_url

class URLLoader(BaseLoader):
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load(self):
        documents = []
        for url in self.urls:
            text = load_pdf_from_url(url)
            documents.extend([Document(page_content=text)])
        return documents
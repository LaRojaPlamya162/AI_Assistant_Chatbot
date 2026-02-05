import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .base import BaseLoader

class PDFDirectoryLoader(BaseLoader):
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load(self) -> List[Document]:
        documents = []
        for file in os.listdir(self.pdf_dir):
            path = os.path.join(self.pdf_dir, file)
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
        return documents
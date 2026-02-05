import arxiv
import trafilatura
import requests
import fitz  # PyMuPDF
from io import BytesIO
from typing import List
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_pdf_url(query):
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    paper = next(search.results())
    return paper.pdf_url, paper.title
  
def load_web_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Failed to fetch URL")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True
    )

    if not text:
        raise ValueError("No main content extracted")

    return text
  
def load_pdf_from_url(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    pdf_bytes = BytesIO(r.content)   # ⚠️ không ghi file
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    if not text.strip():
        raise ValueError("PDF has no extractable text")

    return text
  
def download_pdf(url, save_dir="data/paper"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    filename = url.split("/")[-1] + ".pdf"
    path = Path(save_dir) / filename

    if not path.exists():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)

def replace_abs_to_pdf(url):
    url = url.replace("abs", "pdf")
    return url
  
def split_documents(documents, chunk_size = 1000, overlap = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)
  

import arxiv
import trafilatura
import requests
import fitz  # PyMuPDF
from io import BytesIO
from typing import List
from pathlib import Path
import re

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
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
  
def split_code_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,          # code cần chunk lớn hơn
        chunk_overlap=200,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n\n",
            "\n",
        ],
    )
    return splitter.split_documents(documents)

def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from a given string.

    Args:
        text (str): Input sentence / command.

    Returns:
        List[str]: List of detected URLs.
    """

    URL_PATTERN = re.compile(
    r"""(?xi)
    \b
    (                               # whole url
        (?:https?://|ftp://)        # protocol
        [^\s/$.?#].[^\s]*           # domain + path
    )
    """
)
    if not text:
        return []

    matches = re.findall(URL_PATTERN, text)
    return matches

def extract_clean_text(text: str):
    """
    Extract clean query (for web search) excepts links

    Args:
        Text(str): query for chatbot

    Returns:
        Clean_text(str): clean query (exclude url) for chatbot
    """

    urls = extract_urls(text)
    for url in urls:
        text = text.replace(url,"")
    return text


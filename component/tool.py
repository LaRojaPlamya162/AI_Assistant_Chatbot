import arxiv
import trafilatura
import requests
import fitz  # PyMuPDF
from io import BytesIO
from typing import List
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader

class Tool:
  def __init__(self):
    self.save_dir = "data/papers"

  def get_pdf_url(query):
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    paper = next(search.results())
    return paper.pdf_url, paper.title
  
  def load_web_text(self,url: str) -> str:
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
  
  def load_pdf_from_url(self,url: str) -> str:
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
  
  def download_pdf(self,url, save_dir="data/paper"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    filename = url.split("/")[-1] + ".pdf"
    path = Path(save_dir) / filename

    if not path.exists():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
  
if __name__ == '__main__':
   tool = Tool()
   #print(tool.load_web_text("https://arxiv.org/pdf/1706.03762"))
   #print(tool.load_pdf_from_url(url = ["https://arxiv.org/pdf/1706.03762"]))
   tool.download_pdf(url = "https://arxiv.org/pdf/2602.03301")
   
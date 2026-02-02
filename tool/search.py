from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
class SearchAgent():
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults()
        
    def search_web(self, query: str) -> List[Document]:
        results = self.search_tool.invoke(query)
        docs = []

        for r in results:
            if isinstance(r, dict):
                docs.append(
                    Document(
                        page_content=r.get("snippet", ""),
                        metadata={
                            "title": r.get("title"),
                            "source": r.get("link")
                        }
                    )
                )
        return docs

    def chunking(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

        return splitter.split_documents(docs)

    """def save_docs_to_pdf(self, docs: List[Document], filename: str):
        styles = getSampleStyleSheet()
        story = []

        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", f"Document {i}")
            source = doc.metadata.get("source", "")

            story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
            if source:
                story.append(Paragraph(f"<i>{source}</i>", styles["Normal"]))
            story.append(Spacer(1, 8))

            # tránh lỗi ký tự HTML
            content = doc.page_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(content, styles["Normal"]))
            story.append(Spacer(1, 20))

        doc = SimpleDocTemplate(filename, pagesize=A4)
        doc.build(story)"""

    def answer(self, query: str):
        # 1. Search
        docs = self.search_web(query)

        # 2. Chunk
        chunked_docs = self.chunking(docs)

        # 3. Save PDF
        #os.makedirs("data", exist_ok=True)
        safe_name = sanitize(query)
        pdf_path = f"data/{safe_name}.pdf"
        #self.save_docs_to_pdf(chunked_docs, pdf_path)


def sanitize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_-]", "_", text)
    return text[:5]

if __name__ == "__main__":
    agent = SearchAgent()
    agent.answer("Who is the president of the USA")

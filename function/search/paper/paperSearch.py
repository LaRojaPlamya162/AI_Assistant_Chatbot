import arxiv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from component.loader.url_loader import URLLoader
from component.agent import Agent
from function.search.internet.internetSearch import InternetSearchAgent
class PaperInfoSearchAgent:
    def __init__(self):
        self.client = arxiv.Client()
        self.webAgent = InternetSearchAgent()
    def get_url_link(self, query, k = 1):
        search = arxiv.Search(
            query = query,
            max_results=k,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for r in self.client.results(search):
            results.append(r.pdf_url)
        return results
    
    def infoAnswer(self, question, k = 3):
        search = arxiv.Search(
            query=question,
            max_results=k,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for r in self.client.results(search):
            results.append({
                "title": r.title,
                "pdf_url": r.pdf_url,
                "entry_id": r.entry_id
            })
        print(self.webAgent.answer(question))
        print("Sources: ")
        for res in results:
            print(f"Title: {res['title']}, Link: {res['pdf_url']}") 
        #print(results)

    def fullAnswer(self, question):
        urls = self.get_url_link(question)
        loader = URLLoader(urls = urls)
        agent = Agent(loader)
        print(agent.run(question, task = "paper_review"))

if __name__ == "__main__":
    paper = PaperInfoSearchAgent()
    paper.infoAnswer("What is Soft Actor Critic's algorithm?")

"""

import os
import arxiv
import requests
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

from component.model import Model


class PaperInfoSearch:
    def __init__(self, download_dir="data/papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # ✅ arxiv client (retriever mới)
        self.client = arxiv.Client()
    # -------------------------------------------------
    # 1️⃣ Search paper metadata
    # -------------------------------------------------
    def search_papers(self, query, k=3):
        search = arxiv.Search(
            query=query,
            max_results=k,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for r in self.client.results(search):
            results.append({
                "title": r.title,
                "pdf_url": r.pdf_url,
                "entry_id": r.entry_id
            })

        return results

    # -------------------------------------------------
    # 2️⃣ Download PDF
    # -------------------------------------------------
    def download_pdf(self, url):
        filename = url.split("/")[-1] + ".pdf"
        path = self.download_dir / filename

        if not path.exists():
            print(f"Downloading {filename}...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)

        return str(path)

    # -------------------------------------------------
    # 3️⃣ Build Vector DB from PDFs
    # -------------------------------------------------
    def build_vector_store(self, pdf_paths):
        documents = []

        for pdf in pdf_paths:
            loader = PyMuPDFLoader(pdf)
            documents.extend(loader.load())

        chunks = self.splitter.split_documents(documents)

        vector_store = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        return vector_store

    # -------------------------------------------------
    # 4️⃣ Full RAG Answer
    # -------------------------------------------------
    def fullAnswer(self, question):
        print("🔎 Searching papers...")
        papers = self.search_papers(question)

        pdf_paths = []
        for p in papers:
            print("Found:", p["title"])
            pdf_paths.append(self.download_pdf(p["pdf_url"]))

        print("📚 Building vector store...")
        vector_store = self.build_vector_store(pdf_paths)

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f'''
You are an AI research assistant.
Answer the question using the context below.

Context:
{context}

Question:
{question}
'''

        answer = self.llm.invoke(prompt)
        print("\n===== ANSWER =====\n")
        print(answer)


if __name__ == "__main__":
    paper = PaperInfoSearch()
    paper.fullAnswer("Soft Actor Critic")"""
'''import os
import arxiv
import requests
from pathlib import Path
client = arxiv.Client()
def search_papers(query, k=3):
        search = arxiv.Search(
            query=query,
            max_results=k,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for r in client.results(search):
            results.append({
                "title": r.title,
                "pdf_url": r.pdf_url,
                "entry_id": r.entry_id
            })

        return results

print(search_papers("Soft Actor Critic"))'''
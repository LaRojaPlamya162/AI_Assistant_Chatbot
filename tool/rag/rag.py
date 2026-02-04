from typing import List
import faiss
import requests
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import re
import arxiv
import subprocess
import os

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from component.model import Model
#import trafilatura

class RAGAgent:
    def __init__(self):
        # Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        dim = len(self.embeddings.embed_query("test"))

        # Vector store
        index = faiss.IndexFlatL2(dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.llm = Model()

        self.prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that answers questions using the given context.

Context:
{context}

Question: {question}

If the context does not contain the answer, say "I don't know."
""")

        self._build_knowledge_base()

    # ---------- BUILD KB (ONE TIME) ----------
    def _build_knowledge_base(self):
        #content = 
        pdf_dir = "data/paper"
        files_list = []
        for file_name in os.listdir(pdf_dir):
            full_path = os.path.join(pdf_dir, file_name)
            if os.path.isfile(full_path):
                print(full_path)
                files_list.append(full_path)
        documents = []
        for file in files_list:
            print(f"Load file: {file}")
            loader = PyPDFLoader(file)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)

        print(f"✅ Knowledge base built with {len(chunks)} chunks")

    # ---------- ASK ----------
    def ask(self, question: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)
        prompt = self.prompt.format(
            context=context,
            question=question
        )

        response = self.llm.answer(prompt)

        return response.content


if __name__ == "__main__":
    agent = RAGAgent()
    print(agent.ask("What is Transformer?"))

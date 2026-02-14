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
from component.tool import split_documents, split_code_documents
#import trafilatura

class RAGAgent:
    def __init__(self, loader, source_type: str = "web"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        dim = len(self.embeddings.embed_query("test"))

        index = faiss.IndexFlatL2(dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        self.llm = Model()
        self.loader = loader
        self.prompt = ChatPromptTemplate.from_template( 
            """ You are an AI assistant that answers questions using the given context. You must give the answer as most detail as possible (maximum 1000 words) Context: {context} Question: {question} If the context does not contain the answer, say "I don't know."""
            )
        
        self._build_knowledge_base(source_type)
    def _build_knowledge_base(self, source_type):
        documents = self.loader.load()
        if source_type == "web":
            chunks = split_documents(documents)
        elif source_type == "code":
            chunks = split_code_documents(documents)

        self.vector_store.add_documents(chunks)
        print(f"✅ KB built with {len(chunks)} chunks")
    
    def ask(self, question: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)
        prompt = self.prompt.format(
            context=context,
            question=question
        )

        response = self.llm.answer(prompt)

        return response.content
    
    #agent = RAGAgent()
    #print(agent.ask("What is Transformer?"))

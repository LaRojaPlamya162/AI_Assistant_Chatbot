from typing import List
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader

from component.model import Model
from component.tool import split_documents, split_code_documents
from component.prompt_registry import PromptRegistry

class Agent:
    def __init__(self, loader, source_type: str = "web"):
        self.loader = loader
        self.source_type = source_type

        # ✅ Embedding model (ONLY dùng cho vector DB)
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ LLM (Gemini ↔ Ollama auto fallback)
        self.llm = Model()

        # ✅ Build KB
        self.vector_store = self._build_knowledge_base()

    # ---------------------------------------------------
    # Build Vector Database (SAFE WAY)
    # ---------------------------------------------------
    def _build_knowledge_base(self):
        documents = self.loader.load()

        if self.source_type == "web":
            chunks = split_documents(documents)
        elif self.source_type == "code":
            chunks = split_code_documents(documents)
        else:
            raise ValueError("Unsupported source_type")

        chunks = [c for c in chunks if c.page_content.strip()]

        print(f"🔎 Total clean chunks: {len(chunks)}")

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding
        )

        print("✅ Knowledge Base built successfully")
        return vector_store
    
    def retrieve(self, query: str, k: int = 5) -> str:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        return "\n\n".join(d.page_content for d in docs)


    def run(self, question: str, task: str = "qa"):
        context = self.retrieve(question)

        prompt_template = PromptRegistry.get(task)

        prompt = prompt_template.format(
            context=context,
            question=question
        )

        llm = self.llm.get_llm()
        response = llm.invoke(prompt)

        return response.content

if __name__ == "__main__":
    pass
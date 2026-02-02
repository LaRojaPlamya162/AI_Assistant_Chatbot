from typing import List
import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFium2Loader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from data.content import ContentSources


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

        # LLM
        self.llm = ChatOllama(
            model="qwen2.5:1.5b-instruct",
            base_url="http://localhost:11434",
            temperature=0.1
        )

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
        content = ContentSources()
        pdf_urls: List[str] = content.get_pdf_urls()

        documents = []
        for url in pdf_urls:
            loader = PyPDFium2Loader(f"{url}.pdf")
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

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": docs
        }


if __name__ == "__main__":
    agent = RAGAgent()
    print(agent.ask("What is Transformer?")["answer"])

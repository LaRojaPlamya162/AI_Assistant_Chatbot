# ===== System Lib =====
from typing import List
import os

# ===== Langchain Lib =====
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ===== Project Lib =====
from data.content import ContentSources

class ragAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.embeddings_dim = len(self.embeddings.embed_query("test"))
        self.index = FAISS.IndexFlatL2(self.embeddings_dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            in_memory_docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            in_memory_docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.model = ChatOllama(
            model="qwen2.5:1.5b-instruct", 
            base_url="http://localhost:11434",
            temperature=0.1
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that helps users by providing information from the given context.

{context}

Question:  {question}

Please provide a detailed and accurate answer based on the above context. If the context does not contain relevant information, respond with "I don't know."
""")

  
    def content(self, pdf_urls: List[setattr]):
        #pdf_urls = ContentSources.get_pdf_urls()
        raw_docs = []
        cleaned_docs = []

        for url in pdf_urls:
            loader = PyPDFium2Loader(url)
            raw_docs.extend(loader.load())

        for doc in raw_docs:
            cleaned_text = remove_ununicode_character(doc.page_content)
            cleaned_docs.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )
        return cleaned_docs

    def chunking(self,documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200, ) -> List[Document]:
        """Chunk documents into smaller pieces from better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
        )

        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

        return chunked_docs
    
    def build_vector_store(self, chunked_docs: List[Document]):
        self.vector_store.add_documents(chunked_docs)

    def build_prompt(self, question, docs):
        """
        Convert retrieved docs + question to prompt for LLM
        """
        context = "\n\n".join([f"[Source]\n{d.page_content}" for d in docs])
        return self.prompt.format(context=context, question=question)

    def ask(self, question: str):
        """
        Receive question from user, return answer and sources
        """
        print("Retrieving relevant documents...")
        # Retrieve top-k docs
        self.build_vector_store(
            chunked_docs = self.chunking(
                documents = self.content(
                    pdf_urls=ContentSources.get_pdf_urls()
                    )
                )
            )
        docs = self.vector_store.as_retriever(search_kwargs = {"k": 3}).invoke(question)

        # Build prompt
        prompt_to_llm = self.build_prompt(question, docs)

        response = self.llm.invoke(prompt_to_llm)

        return {
            "answer": response.content,
            "sources": docs
        }
    

def remove_ununicode_character(text: str) -> str:
    # Only keep ASCII characters
    return ''.join([char if ord(char) < 128 else ' ' for char in text])

if __name__ == "__main__":
    agent = ragAgent()
    agent.ask("How the weather like today?")
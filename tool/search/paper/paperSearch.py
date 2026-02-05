from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss

from component.model import Model
from tool.rag.loader.url_loader import URLLoader
from tool.rag.rag import RAGAgent


class PaperInfoSearch:
    def __init__(self):
        self.retriever = ArxivRetriever(
            load_max_docs=3,
            sort_by="relevance"
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an AI assistant that answers questions using the given context.
You must give the answer as most detail as possible (maximum 1000 words)

Context:
{context}

Question: {question}

If the context does not contain the answer, say "I don't know."
"""
        )

        # ✅ Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    def infoAnswer(self, question):
        docs = self.retriever.invoke(question)
        for doc in docs:
            print(doc.metadata["Entry ID"])

    def fullAnswer(self, question):
        docs = self.retriever.invoke(question)
        urls = [doc.metadata['Entry ID'] for doc in docs]
        loader = URLLoader(urls = urls)
        agent = RAGAgent(loader)
        print(agent.ask(question))

if __name__ == "__main__":
    paper = PaperInfoSearch()
    paper.fullAnswer("Soft Actor Critic")
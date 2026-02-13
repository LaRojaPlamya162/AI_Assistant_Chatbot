from typing import List, Callable, Optional
from pathlib import Path

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from tool.rag.rag import RAGAgent
from tool.rag.loader.repo_loader import repoLoader
class RepoSearch:
    """
    Load Git repository -> split -> return LangChain Documents ready for RAG.
    """
    def __init__(self):
        pass
    def answer(self, 
               clone_url: str,
               question: str
               ):
        loader = repoLoader(clone_url = clone_url)
        #print(loader.load())
        agent = RAGAgent(loader, source_type = "code")
        print(agent.ask(question))

if __name__ == '__main__':
    search = RepoSearch()
    search.answer(clone_url = "https://github.com/hieudz2k4/AI", question = "What is the main content of this repo ?")
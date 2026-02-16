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
from component.loader.pdf_loader import PDFDirLoader
from component.agent import Agent
class RAGAgent:
    def __init__(self):
        self.loader = PDFDirLoader()

    def ask(self, question):
        agent = Agent(self.loader)
        return agent.run(question, task = "qa")
    
if __name__ == "__main__":
    agent = RAGAgent()
    print(agent.ask("Who is the 40th president of USA"))


                                   
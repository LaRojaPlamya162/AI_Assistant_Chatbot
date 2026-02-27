from typing import List
from component.engine import Engine
from component.loader.url_loader import URLLoader
class WebSearchAgent:
    """
    Load Git repository -> split -> return LangChain Documents ready for RAG.
    """
    def __init__(self):
        pass

    def answer(self, 
               clone_url: List[str],
               question: str
               ):
        loader = URLLoader(clone_url = clone_url)
        #print(loader.load())
        agent = Engine(loader, source_type = "code")
        print(agent.run(question, task = "code_explain"))

if __name__ == '__main__':
    search = WebSearchAgent()
    search.answer(clone_url = "https://github.com/hieudz2k4/AI", question = "What is the main content of this repo ?")
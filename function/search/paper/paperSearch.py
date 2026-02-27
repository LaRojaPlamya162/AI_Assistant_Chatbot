import arxiv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from component.loader.url_loader import URLLoader
from component.engine import Engine
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
        agent = Engine(loader)
        print(agent.run(question, task = "paper_review"))

if __name__ == "__main__":
    paper = PaperInfoSearchAgent()
    paper.infoAnswer("What is Soft Actor Critic's algorithm?")

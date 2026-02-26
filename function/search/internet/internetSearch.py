from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import SearxSearchWrapper, GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

class InternetSearchAgent:
    def __init__(self):
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

        # Init tools
        self.google_serper = GoogleSerperAPIWrapper()
        self.tavily = TavilySearch(k=3)
        self.duckduckgo = DuckDuckGoSearchResults()
        self.searx = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

        # Priority order
        self.models = [
            ("Google Serper", self.google_serper),
            ("Tavily", self.tavily),
            ("DuckDuckGo", self.duckduckgo),
            ("Searx", self.searx),
        ]

    def choose_model(self, query: str):
        for name, model in self.models:
            try:
                print(f"🔎 Trying {name}...")
                result = model.run(query)

                if result and result.strip():
                    print(f"✅ Selected: {name}")
                    return name, result

            except Exception as e:
                print(f"❌ {name} failed: {e}")

        raise RuntimeError("All web search tools failed")

    def answer(self, query: str):
        tool_name, docs = self.choose_model(query)
        #print("\n=== RESULT ===")
        print(f"Source: {tool_name}")
        print(docs)


if __name__ == "__main__":
    agent = InternetSearchAgent()
    agent.answer( "Can you speak Vietnamese ? May i ask you by Vietnamese? (answer with max 100 words)")
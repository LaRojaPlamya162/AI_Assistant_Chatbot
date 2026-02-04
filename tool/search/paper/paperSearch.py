from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import ArxivAPIWrapper
from component.model import Model
class PaperInfoSearch:
    def __init__(self):
        self.retriever = ArxivRetriever(
            load_max_docs = 5,
            sort_by = "relevance"
        )
        self.arxiv = ArxivAPIWrapper()
        self.prompt = ChatPromptTemplate.from_template('''
You are an AI assistant that answers questions using the given context. You must give the answer as most detail as possible (maximum 1000 words)

Context:
{context}

Question: {question}

If the context does not contain the answer, say "I don't know."
''')
        self.llm = Model()

    def infoAnswer(self, question):
        #docs = self.retriever.invoke(query)
        docs = self.arxiv.load(question)
        for doc in docs:
            print(doc)
    
    def fullAnswer(self, question):
        docs = self.arxiv.load(question)
        #docs = self.retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = self.prompt.format(
            context = context,
            question = question
        )
        response = self.llm.answer(prompt)
        print("Answer: ", response.content)
        #print("Sources: ", docs)

if __name__ == '__main__':
    paper = PaperInfoSearch()
    paper.fullAnswer("Behavior Cloning")

"""import arxiv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from component.model import Model
from langchain_community.utilities import ArxivAPIWrapper
# 🔥 FIX HTTP 301
arxiv.arxiv.API_URL = "https://export.arxiv.org/api/query"


class PaperInfoSearch:
    def __init__(self):
        self.client = arxiv.Client()
        self.llm = Model()

        self.prompt = ChatPromptTemplate.from_template('''
You are an AI assistant that answers questions using the given context.
You must give the answer as most detail as possible (maximum 1000 words).

Context:
{context}

Question: {question}

If the context does not contain the answer, say "I don't know."
''')

    def _search(self, query, max_results=5):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        docs = []
        for paper in self.client.results(search):
            docs.append(
                Document(
                    page_content=paper.summary,
                    metadata={
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "published": paper.published.isoformat(),
                        "entry_id": paper.entry_id,
                        "pdf_url": paper.pdf_url,
                    }
                )
            )
        return docs

    def infoAnswer(self, query):
        docs = self._search(query)
        for doc in docs:
            print(doc)

    def fullAnswer(self, question):
        docs = self._search(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = self.prompt.format(
            context=context,
            question=question
        )

        response = self.llm.answer(prompt)
        print("Answer:", response.content)


if __name__ == "__main__":
    paper = PaperInfoSearch()
    paper.infoAnswer("Soft Actor Critic")"""

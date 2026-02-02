from langchain_community.retrievers import ArxivRetriever
class PaperInfoSearch:
    def __init__(self):
        self.retriever = ArxivRetriever(
            load_max_docs = 5,
            sort_by = "relevance"
        )
    def query(self, doc):
        print('\n'f'Title: {doc.metadata["Title"]}')
        print(f'Content: {doc.page_content}')
        print(f'Published Date: {doc.metadata["Published"]}')
        print(f'Authors: {doc.metadata["Authors"]}')
        

    def answer(self, query):
        docs = self.retriever.invoke(query)
        for doc in docs:
            self.query(doc)
        
if __name__ == '__main__':
    paper = PaperInfoSearch()
    paper.answer("Soft Actor Critic")
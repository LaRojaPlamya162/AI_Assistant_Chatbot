from langchain_core.documents import Document
class CodeAnalysisLoader:
  def __init__(self, code: str):
    self.code = code
  def load(self):
    documents = [
    Document(
        page_content=self.code,
        metadata={"source": "user_input"}
    )
]
    return documents



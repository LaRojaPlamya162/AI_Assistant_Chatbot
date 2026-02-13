from typing import List, Callable, Optional
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from .base import BaseLoader
class repoLoader(BaseLoader):
  def __init__(self,
               clone_url: str, 
               repo_path: str = "./repo",
               branch: str = "main",
               file_filter: Optional[Callable[[str], bool]] = None):
    self.clone_url = clone_url
    self.repo_path = repo_path
    self.branch = branch
    if file_filter is None:
            self.file_filter = self._default_file_filter
    else:
            self.file_filter = file_filter
    self.loader = GitLoader(
            clone_url = clone_url,
            repo_path=str(repo_path),
            branch = branch,
            file_filter = file_filter
        )
  def _default_file_filter(self, path: str) -> bool:
    # chỉ index file mang thông tin hệ thống
    allowed_ext = (".py", ".md")

    if not path.endswith(allowed_ext):
        return False

    ignore_keywords = [
        "download",
        "data_source",
        ".gitkeep",
        "requirements",
        "docker",
        "__pycache__"
    ]

    if any(k in path.lower() for k in ignore_keywords):
        return False

    return True
    
  def load(self):
    documents = self.loader.load()
    for doc in documents:
            source = doc.metadata.get("source", "")
            doc.metadata = {
                "file_path": source,
                "file_name": Path(source).name,
                "repo": self.clone_url,
            }
    return documents
    
if __name__ == "__main__":
     pass
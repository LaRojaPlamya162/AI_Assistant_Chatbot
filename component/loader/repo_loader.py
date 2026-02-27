# from typing import List, Callable, Optional
# from pathlib import Path
# from langchain_community.document_loaders import GitLoader
# from .base import BaseLoader
# class repoLoader(BaseLoader):
#   def __init__(self,
#                clone_url: List[str], 
#                repo_path: str = "./repo",
#                branch: str = "main",
#                file_filter: Optional[Callable[[str], bool]] = None):
#     self.clone_url = clone_url
#     self.repo_path = repo_path
#     self.branch = branch
#     if file_filter is None:
#             self.file_filter = self._default_file_filter
#     else:
#             self.file_filter = file_filter
#     self.loader = []
#     for url in clone_url:
#          load = GitLoader(
#             clone_url = url,
#             repo_path=str(repo_path),
#             branch = branch,
#             file_filter = file_filter
#         )
#          self.loader.extend(load)
        
#   def _default_file_filter(self, path: str) -> bool:
#     # chỉ index file mang thông tin hệ thống
#     allowed_ext = (".py", ".md")

#     if not path.endswith(allowed_ext):
#         return False

#     ignore_keywords = [
#         "download",
#         "data_source",
#         ".gitkeep",
#         "requirements",
#         "docker",
#         "__pycache__"
#     ]

#     if any(k in path.lower() for k in ignore_keywords):
#         return False

#     return True
    
#   def load(self):
#     documents = self.loader.load()
#     for doc in documents:
#             source = doc.metadata.get("source", "")
#             doc.metadata = {
#                 "file_path": source,
#                 "file_name": Path(source).name,
#                 "repo": self.clone_url,
#             }
#     return documents
    
# if __name__ == "__main__":
#      pass

from typing import List, Callable, Optional
from pathlib import Path
import os
import requests
from langchain_community.document_loaders import GitLoader
from .base import BaseLoader


class repoLoader(BaseLoader):
    def __init__(
        self,
        clone_url: List[str],                  # MULTI REPO
        repo_root: str = "./repo_cache",       # ROOT CACHE FOLDER
        branch: Optional[str] = None,          # allow override
        file_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.clone_urls = clone_url if isinstance(clone_url, list) else [clone_url]
        self.repo_root = repo_root
        self.branch_override = branch

        os.makedirs(self.repo_root, exist_ok=True)

        self.file_filter = file_filter or self._default_file_filter

        self.documents = []

        self._load_repositories()

    # --------------------------------------------------
    # Detect default branch from GitHub
    # --------------------------------------------------
    def _detect_default_branch(self, url: str) -> str:
        if self.branch_override:
            return self.branch_override

        try:
            parts = url.replace(".git", "").split("/")
            owner, repo = parts[-2], parts[-1]

            api = f"https://api.github.com/repos/{owner}/{repo}"
            r = requests.get(api, timeout=10)

            if r.status_code == 200:
                branch = r.json().get("default_branch", "main")
                print(f"[INFO] Detected branch '{branch}' for {repo}")
                return branch

        except Exception as e:
            print(f"[WARN] Branch detect failed: {e}")

        return "main"

    # --------------------------------------------------
    # Get local path for each repo
    # --------------------------------------------------
    def _repo_local_path(self, url: str) -> str:
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        return os.path.join(self.repo_root, repo_name)

    # --------------------------------------------------
    # Clone (if needed) + Load documents
    # --------------------------------------------------
    def _load_repositories(self):
        for url in self.clone_urls:
            local_path = self._repo_local_path(url)
            branch = self._detect_default_branch(url)

            already_cloned = os.path.exists(local_path) and os.listdir(local_path)

            if already_cloned:
                print(f"[CACHE] Using cached repo: {local_path}")
            else:
                print(f"[CLONE] {url} → {local_path}")

            loader = GitLoader(
                clone_url=url,
                repo_path=local_path,
                branch=branch,
                file_filter=self.file_filter,
            )

            try:
                repo_docs = loader.load()
            except Exception as e:
                print(f"[ERROR] Failed loading {url}: {e}")
                continue

            for doc in repo_docs:
                source = doc.metadata.get("source", "")
                doc.metadata = {
                    "file_path": source,
                    "file_name": Path(source).name,
                    "repo": url,
                }

            self.documents.extend(repo_docs)

        print(f"[INFO] Loaded {len(self.documents)} documents from {len(self.clone_urls)} repos")

    # --------------------------------------------------
    # Default file filter (important for RAG quality)
    # --------------------------------------------------
    def _default_file_filter(self, path: str) -> bool:
        allowed_ext = (".py", ".md", ".ipynb")

        if not path.endswith(allowed_ext):
            return False

        ignore_keywords = [
            "dataset",
            "data",
            "__pycache__",
            ".git",
            "node_modules",
            "build",
            "dist",
            "logs",
            "checkpoint",
        ]

        if any(k in path.lower() for k in ignore_keywords):
            return False

        return True

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def load(self):
        return self.documents
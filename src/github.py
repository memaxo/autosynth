import os
import logging
from typing import List, Optional
from pathlib import Path

from github import Github, Repository
from tqdm import tqdm

from langchain.schema import Document

logger = logging.getLogger(__name__)

class RepoCollector:
    """
    A dedicated module for extracting and formatting repository contents
    from GitHub. Reuses logic adapted from repototxt.py, but integrated
    into AutoSynth's structure.
    """
    def __init__(self, github_token: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("Missing GitHub token. Please set GITHUB_TOKEN env variable or pass it to RepoCollector.")
        self.github_client = Github(self.github_token)
        self.cache_dir = cache_dir
        
        # Commonly skipped binary and large file extensions
        self.binary_extensions = [
            '.exe', '.dll', '.so', '.zip', '.tar', '.rar', '.7z', '.pdf',
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.png', '.jpg',
            '.jpeg', '.gif', '.mp3', '.mp4', '.db', '.jar', '.class', '.war',
            '.pyc', '.whl', '.svg', '.ico', '.msi', '.dmg', '.lock',
            '.otf', '.ttf', '.woff', '.woff2', '.keystore', '.jks',
            '.crt', '.key', '.pfx', '.pem', '.pub', '.nupkg', '.snupkg',
            '.git', '.gitignore', '.DS_Store'
        ]
        
    def collect(self, repo_url: str) -> List[Document]:
        """
        Retrieves and structures the content of a public GitHub repo
        into a list of Documents for downstream processing.

        Returns:
            A list of Documents containing:
            - README content (if available)
            - File listings (repository structure)
            - Extracted text content of non-binary files
        """
        logger.info(f"Collecting GitHub repo contents: {repo_url}")
        
        # 1. Handle caching (if desired)
        docs = self._check_cache(repo_url)
        if docs is not None:
            logger.info(f"Using cached contents for repository: {repo_url}")
            return docs
        
        # 2. Fetch repository from GitHub
        repo_path = repo_url.replace("https://github.com/", "")
        repo = self.github_client.get_repo(repo_path)
        
        # 3. Extract README content
        readme_content = self._get_readme(repo)
        
        # 4. Traverse and extract the repository structure + file contents
        repo_structure, repo_documents = self._traverse_repo_iteratively(repo)
        
        # 5. Combine them into a single list of Documents
        all_docs = []
        
        # (a) README as a Document
        if readme_content:
            all_docs.append(Document(
                page_content=readme_content,
                metadata={"source_url": repo_url, "file_path": "README.md", "content_type": "github_readme"}
            ))
        
        # (b) Repository Structure as a single Document
        all_docs.append(Document(
            page_content=repo_structure,
            metadata={"source_url": repo_url, "file_path": "REPO_STRUCTURE", "content_type": "github_structure"}
        ))
        
        # (c) Individual file documents
        all_docs.extend(repo_documents)
        
        # 6. Cache results (if configured)
        self._cache_documents(repo_url, all_docs)
        
        return all_docs

    def _get_readme(self, repo: Repository.Repository) -> str:
        """
        Retrieve the content of the README.md if available.
        """
        try:
            readme = repo.get_contents("README.md")
            return readme.decoded_content.decode("utf-8")
        except Exception:
            return ""

    def _traverse_repo_iteratively(self, repo: Repository.Repository):
        """
        Traverse the repository iteratively to avoid recursion limits
        for large repositories, skipping known binary file types.
        """
        repo_structure = [f"Repository: {repo.full_name}"]
        docs = []
        
        dirs_to_visit = [("", repo.get_contents(""))]
        visited = set()
        
        while dirs_to_visit:
            path, contents = dirs_to_visit.pop()
            visited.add(path)
            for content in tqdm(contents, desc=f"Processing {path}", leave=False):
                if content.type == "dir":
                    if content.path not in visited:
                        repo_structure.append(f"{path}/{content.name}/")
                        dirs_to_visit.append((f"{path}/{content.name}", repo.get_contents(content.path)))
                else:
                    # Build the structure line
                    file_line = f"{path}/{content.name}"
                    repo_structure.append(file_line)
                    
                    # Skip if extension is listed as binary
                    if any(content.name.lower().endswith(ext) for ext in self.binary_extensions):
                        continue
                    
                    # Attempt to decode content
                    file_text = self._fetch_file_text(content)
                    if file_text:
                        docs.append(Document(
                            page_content=file_text,
                            metadata={
                                "source_url": f"https://github.com/{repo.full_name}",
                                "file_path": file_line.strip("/"),
                                "content_type": "github_file"
                            }
                        ))
        
        return "\n".join(repo_structure), docs

    def _fetch_file_text(self, content) -> str:
        """
        Decode file content safely, skipping files that fail decoding.
        """
        try:
            decoded = content.decoded_content.decode("utf-8")
            return decoded
        except UnicodeDecodeError:
            try:
                return content.decoded_content.decode("latin-1")
            except UnicodeDecodeError:
                logger.warning(f"Skipped file {content.path} due to unsupported encoding.")
                return ""
        except Exception as e:
            logger.warning(f"Error fetching file {content.path}: {e}")
            return ""
    
    def _check_cache(self, repo_url: str) -> Optional[List[Document]]:
        """
        Check local cache for previously extracted Documents.
        """
        if not self.cache_dir:
            return None
        import pickle
        cache_file = self.cache_dir / f"{hash(repo_url)}_repo.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed for {repo_url}: {e}")
                return None
        return None
    
    def _cache_documents(self, repo_url: str, docs: List[Document]) -> None:
        """
        Cache the fetched Documents for future reuse.
        """
        if not self.cache_dir:
            return
        import pickle
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{hash(repo_url)}_repo.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(docs, f)
        except Exception as e:
            logger.warning(f"Failed to write cache for {repo_url}: {e}")

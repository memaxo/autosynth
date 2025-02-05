from .config import CONFIG
from .providers.brave import BraveSearchAPIWrapper
from .providers.exa import ExaAPIWrapper
from .providers.duckduckgo import DuckDuckGoSearchAPIWrapper
from .providers.google import GoogleSearchAPIWrapper
from .providers.tavily import TavilySearchAPIWrapper
from .collector import Collector
from .processor import Processor
from .generator import ContentGenerator
from .search import Search
from .db import ProjectStateDB
from .pipeline import Project

def create_rate_limiter(default_rps: float, max_bucket: int):
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(requests_per_second=default_rps, max_bucket_size=max_bucket)

def create_providers():
    rate_limiter = create_rate_limiter(CONFIG.RATE_LIMIT_CONFIG.REQUESTS_PER_SECOND,
                                       CONFIG.RATE_LIMIT_CONFIG.MAX_BUCKET_SIZE)
    providers = {
        "brave": BraveSearchAPIWrapper(api_key=CONFIG.API_CONFIG.BRAVE_API_KEY,
                                         rate_limiter=rate_limiter),
        "exa": ExaAPIWrapper(api_key=CONFIG.API_CONFIG.EXA_API_KEY,
                             rate_limiter=rate_limiter),
        "duckduckgo": DuckDuckGoSearchAPIWrapper(rate_limiter=rate_limiter),
        "google": GoogleSearchAPIWrapper(api_key=CONFIG.API_CONFIG.GOOGLE_API_KEY,
                                         cse_id=CONFIG.API_CONFIG.GOOGLE_CSE_ID,
                                         rate_limiter=rate_limiter),
        "tavily": TavilySearchAPIWrapper(api_key=CONFIG.API_CONFIG.TAVILY_API_KEY,
                                         rate_limiter=rate_limiter)
    }
    return providers

def create_collector():
    return Collector(
        cache_dir=CONFIG.PATH_CONFIG.CACHE_DIR,
        rate_limit=CONFIG.RATE_LIMIT_CONFIG.DEFAULT_RATE_LIMIT,
        chunk_size=CONFIG.DEFAULT_PARAMS.CHUNK_SIZE,
        chunk_overlap=CONFIG.DEFAULT_PARAMS.CHUNK_OVERLAP,
    )

def create_processor(validator=None):
    return Processor(
        validator=validator
    )

def create_generator():
    return ContentGenerator(
        model_name=CONFIG.MODEL_CONFIG.MODEL_PATH,
        max_retries=CONFIG.PIPELINE_CONFIG.GENERATE["max_retries"],
        completion_window="1h"
    )

def create_search(validator=None, db=None):
    providers = create_providers()
    return Search(
        providers=providers,
        validator=validator,
        db=db
    )

def create_state_db():
    return ProjectStateDB()

def create_project(project_id: str, topic: str):
    return Project(
        project_id,
        topic,
        base_path=CONFIG.PATH_CONFIG.PROJECT_DIR,
        searcher=create_search(),
        repo_collector=None,  # Optionally, instantiate RepoCollector externally if needed.
        collector=create_collector(),
        processor=create_processor(),
        generator=create_generator(),
        state_db=create_state_db()
    )
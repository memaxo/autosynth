import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class APIConfig:
    BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

@dataclass
class RateLimitConfig:
    REQUESTS_PER_SECOND: float = 1.0
    MAX_BUCKET_SIZE: int = 5
    DEFAULT_RATE_LIMIT: float = 1.0

@dataclass
class DefaultParams:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TIMEOUT: int = 30
    MAX_TOKENS: int = 2000
    MAX_WORKERS: int = 5
    MAX_CONCURRENCY: int = 5
    EMBEDDING_BATCH_SIZE: int = 32
    MAX_TEXT_LENGTH: int = 512
    EMBEDDING_DIMENSION: int = 128
    SIMHASH_THRESHOLD: int = 3
    CACHE_SIZE: int = 10000
    MIN_ENGLISH_RATIO: float = 0.5
    PREVIEW_LENGTH: int = 1000
    DEFAULT_MIN_QUALITY: float = 0.7
    DEFAULT_BATCH_SIZE: int = 5

@dataclass
class PathConfig:
    DB_DIR: Path = Path(os.getenv("DB_DIR", "~/.autosynth/db")).expanduser()
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./cache")).expanduser()
    PROJECT_DIR: Path = Path(os.getenv("PROJECT_DIR", "./")).expanduser()

@dataclass
class ModelConfig:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/Volumes/JMD/models/phi4-4bit")
    MAX_TOKENS: int = 100
    TEMPERATURE: float = 0.1

@dataclass
class PipelineConfig:
    DISCOVER: dict = field(default_factory=lambda: {
        "web_batch_size": 50,
        "github_batch_size": 20,
        "min_quality_score": 0.7
    })
    COLLECT: dict = field(default_factory=lambda: {
        "max_tokens_per_doc": 2000,
        "timeout": 30,
        "github_max_files": 10
    })
    PROCESS: dict = field(default_factory=lambda: {
        "chunk_size": 1000,
        "overlap": 200
    })
    GENERATE: dict = field(default_factory=lambda: {
        "batch_size": 5,
        "max_retries": 3
    })

@dataclass
class SearchConfig:
    ENABLED_PROVIDERS: list = field(default_factory=lambda: os.getenv("ENABLED_PROVIDERS", "google").split(","))
    DEFAULT_PROVIDER: str = "google"
    FALLBACK_PROVIDERS: list = field(default_factory=lambda: ["google"])
    MAX_RESULTS_PER_PROVIDER: int = 10
    MAX_CACHE_AGE_HOURS: int = 24
    DEFAULT_NUM_RESULTS: int = 10
    POSITION_WEIGHT: float = 2.0
    ACADEMIC_BOOST: float = 1.25
    MIN_QUALITY_SCORE: float = 0.7

@dataclass
class ValidatorConfig:
    MAX_RECENT_FINGERPRINTS: int = 1000
    ACADEMIC_PATTERNS: dict = field(default_factory=lambda: {
        "arxiv": r'arxiv\.org/abs/\d+\.\d+',
        "doi": r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
        "academic_keywords": r'\b(thesis|dissertation|journal|conference|proceedings|research|study)\b',
        "citations": r'\[\d+\]|\(\w+\s+et\s+al\.,\s+\d{4}\)'
    })
    ACADEMIC_DOMAINS: list = field(default_factory=lambda: [
        "arxiv.org", "github.com", "*.edu", "*.ac.*", "scholar.google.com", "semanticscholar.org"
    ])
    BLOCKED_URL_PATTERNS: list = field(default_factory=lambda: [
        '/ads/', '/tracking/', '/analytics/', '/sponsored/', '/advertisement/'
    ])
    PREFERRED_URL_PATTERNS: list = field(default_factory=lambda: [
        '/paper/', '/article/', '/research/', '/publication/', '/doc/', '/pdf/'
    ])

@dataclass
class Config:
    API_CONFIG: APIConfig = APIConfig()
    RATE_LIMIT_CONFIG: RateLimitConfig = RateLimitConfig()
    DEFAULT_PARAMS: DefaultParams = DefaultParams()
    PATH_CONFIG: PathConfig = PathConfig()
    MODEL_CONFIG: ModelConfig = ModelConfig()
    PIPELINE_CONFIG: PipelineConfig = PipelineConfig()
    SEARCH_CONFIG: SearchConfig = SearchConfig()
    VALIDATOR_CONFIG: ValidatorConfig = ValidatorConfig()

def load_external_config(file_path: str = "config.yaml") -> dict:
    try:
        with open(file_path, "r") as f:
            external_config = yaml.safe_load(f)
            return external_config or {}
    except FileNotFoundError:
        return {}

def merge_dicts(default: dict, external: dict) -> dict:
    result = default.copy()
    for key, value in external.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def get_config() -> Config:
    external_config = load_external_config()
    cfg = Config()
    if "API_CONFIG" in external_config:
        cfg.API_CONFIG = APIConfig(**merge_dicts(cfg.API_CONFIG.__dict__, external_config.get("API_CONFIG", {})))
    if "RATE_LIMIT_CONFIG" in external_config:
        cfg.RATE_LIMIT_CONFIG = RateLimitConfig(**merge_dicts(cfg.RATE_LIMIT_CONFIG.__dict__, external_config.get("RATE_LIMIT_CONFIG", {})))
    if "DEFAULT_PARAMS" in external_config:
        cfg.DEFAULT_PARAMS = DefaultParams(**merge_dicts(cfg.DEFAULT_PARAMS.__dict__, external_config.get("DEFAULT_PARAMS", {})))
    if "PATH_CONFIG" in external_config:
        path_config = merge_dicts({k: str(v) for k, v in cfg.PATH_CONFIG.__dict__.items()}, external_config.get("PATH_CONFIG", {}))
        cfg.PATH_CONFIG = PathConfig(
            DB_DIR=Path(path_config.get("DB_DIR", str(cfg.PATH_CONFIG.DB_DIR))),
            CACHE_DIR=Path(path_config.get("CACHE_DIR", str(cfg.PATH_CONFIG.CACHE_DIR))),
            PROJECT_DIR=Path(path_config.get("PROJECT_DIR", str(cfg.PATH_CONFIG.PROJECT_DIR)))
        )
    if "MODEL_CONFIG" in external_config:
        cfg.MODEL_CONFIG = ModelConfig(**merge_dicts(cfg.MODEL_CONFIG.__dict__, external_config.get("MODEL_CONFIG", {})))
    if "PIPELINE_CONFIG" in external_config:
        cfg.PIPELINE_CONFIG = PipelineConfig(**merge_dicts(cfg.PIPELINE_CONFIG.__dict__, external_config.get("PIPELINE_CONFIG", {})))
    if "SEARCH_CONFIG" in external_config:
        cfg.SEARCH_CONFIG = SearchConfig(**merge_dicts(cfg.SEARCH_CONFIG.__dict__, external_config.get("SEARCH_CONFIG", {})))
    if "VALIDATOR_CONFIG" in external_config:
        cfg.VALIDATOR_CONFIG = ValidatorConfig(**merge_dicts(cfg.VALIDATOR_CONFIG.__dict__, external_config.get("VALIDATOR_CONFIG", {})))
    return cfg

# Global configuration instance
CONFIG = get_config()
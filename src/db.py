import os
import aiosqlite
import logging
import numpy as np
import asyncio
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from .config import CONFIG

from langchain_community.vectorstores import SQLiteVec
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ranking.embeddings import EmbeddingGenerator
from functools import wraps
from .utils.hashing import content_hash, url_hash, fingerprint_hash

logger = logging.getLogger(__name__)

# Database configuration
try:
    DEFAULT_DB_DIR = CONFIG.PATH_CONFIG.DB_DIR
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create default DB directory: {e}")
    DEFAULT_DB_DIR = Path(".")

DB_FILES = {
    "search": DEFAULT_DB_DIR / "search.db",
    "validation": DEFAULT_DB_DIR / "validation.db",
    "embeddings": DEFAULT_DB_DIR / "embeddings.db",
    "cache": DEFAULT_DB_DIR / "cache.db",
    "project": DEFAULT_DB_DIR / "project.db"  # For project state management
}

# Table configuration
DEFAULT_TABLES = {
    "search": "search_cache",
    "validation": "validation_embeddings",
    "embeddings": "embeddings_store",
    "cache": "cache_store",
    "project": "project_state"
}

class ProjectStateDB:
    """
    Asynchronous database manager for project state storage.
    """

    def __init__(self, db_type: str = "project"):
        if db_type not in DB_FILES:
            raise ValueError(f"Invalid db_type. Must be one of: {list(DB_FILES.keys())}")
        self.db_file = str(DB_FILES[db_type])
        self.table_name = DEFAULT_TABLES[db_type]
        self._conn = None

    async def get_connection(self):
        if self._conn is None:
            try:
                self._conn = await aiosqlite.connect(self.db_file)
            except Exception as e:
                logger.error(f"Error connecting to database {self.db_file}: {e}")
                raise
        return self._conn

    async def _init_tables(self):
        conn = await self.get_connection()
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    last_stage TEXT DEFAULT 'init',
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    config JSON
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS discovered_urls (
                    project_id TEXT,
                    url TEXT,
                    discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id),
                    UNIQUE(project_id, url)
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS github_repos (
                    project_id TEXT,
                    repo_data JSON,
                    discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS collected_docs (
                    project_id TEXT,
                    doc_data JSON,
                    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_docs (
                    project_id TEXT,
                    doc_data JSON,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS generated_docs (
                    project_id TEXT,
                    doc_data JSON,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    project_id TEXT,
                    metric_name TEXT,
                    current_value INTEGER,
                    target_value INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id),
                    UNIQUE(project_id, metric_name)
                )
            """)
        await conn.commit()

    async def create_project(self, project_id: str, topic: str, config: Dict) -> None:
        conn = await self.get_connection()
        await conn.execute(
            "INSERT INTO projects (project_id, topic, config) VALUES (?, ?, ?)",
            (project_id, topic, json.dumps(config))
        )
        await conn.commit()

    async def get_project(self, project_id: str) -> Optional[Dict]:
        conn = await self.get_connection()
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)) as cursor:
            result = await cursor.fetchone()
        if result:
            return {
                "project_id": result["project_id"],
                "topic": result["topic"],
                "last_stage": result["last_stage"],
                "last_updated": result["last_updated"],
                "config": json.loads(result["config"])
            }
        return None

    async def update_project_state(self, project_id: str, last_stage: str) -> None:
        conn = await self.get_connection()
        try:
            await conn.execute(
                """UPDATE projects 
                   SET last_stage = ?, last_updated = CURRENT_TIMESTAMP
                   WHERE project_id = ?""",
                (last_stage, project_id)
            )
            await conn.commit()
        except Exception as e:
            logger.error(f"Error updating project state for project_id {project_id}: {e}")
            raise

    async def add_urls(self, project_id: str, urls: List[str]) -> None:
        conn = await self.get_connection()
        await conn.executemany(
            "INSERT OR IGNORE INTO discovered_urls (project_id, url) VALUES (?, ?)",
            [(project_id, url) for url in urls]
        )
        await conn.commit()

    async def get_urls(self, project_id: str) -> List[str]:
        conn = await self.get_connection()
        async with conn.execute("SELECT url FROM discovered_urls WHERE project_id = ?", (project_id,)) as cursor:
            results = await cursor.fetchall()
        return [r[0] for r in results]

    async def add_github_repos(self, project_id: str, repos: List[Dict]) -> None:
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.executemany(
                "INSERT INTO github_repos (project_id, repo_data) VALUES (?, ?)",
                [(project_id, json.dumps(repo)) for repo in repos]
            )
            await conn.commit()

    async def get_github_repos(self, project_id: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_file) as conn:
            async with conn.execute("SELECT repo_data FROM github_repos WHERE project_id = ?", (project_id,)) as cursor:
                results = await cursor.fetchall()
        return [json.loads(r[0]) for r in results]

    async def update_metrics(self, project_id: str, metric_name: str, current: int, target: int) -> None:
        conn = await self.get_connection()
        try:
            await conn.execute(
                """INSERT OR REPLACE INTO metrics (project_id, metric_name, current_value, target_value)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(project_id, metric_name) DO UPDATE SET
                   current_value = ?, target_value = ?, updated_at = CURRENT_TIMESTAMP""",
                (project_id, metric_name, current, target, current, target)
            )
            await conn.commit()
        except Exception as e:
            logger.error(f"Error updating metrics for project_id {project_id}, metric {metric_name}: {e}")
            raise

    async def get_metrics(self, project_id: str) -> Dict[str, Dict[str, int]]:
        async with aiosqlite.connect(self.db_file) as conn:
            async with conn.execute("SELECT metric_name, current_value, target_value FROM metrics WHERE project_id = ?", (project_id,)) as cursor:
                results = await cursor.fetchall()
        return {
            r[0]: {"current": r[1], "target": r[2]}
            for r in results
        }

    async def add_collected_doc(self, project_id: str, doc_dict: Dict) -> None:
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.execute(
                "INSERT INTO collected_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
            await conn.commit()

    async def add_processed_doc(self, project_id: str, doc_dict: Dict) -> None:
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.execute(
                "INSERT INTO processed_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
            await conn.commit()

    async def add_generated_doc(self, project_id: str, doc_dict: Dict) -> None:
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.execute(
                "INSERT INTO generated_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
            await conn.commit()

    async def get_collected_docs(self, project_id: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_file) as conn:
            async with conn.execute("SELECT doc_data FROM collected_docs WHERE project_id = ?", (project_id,)) as cursor:
                results = await cursor.fetchall()
        return [json.loads(r[0]) for r in results]

    async def get_processed_docs(self, project_id: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_file) as conn:
            async with conn.execute("SELECT doc_data FROM processed_docs WHERE project_id = ?", (project_id,)) as cursor:
                results = await cursor.fetchall()
        return [json.loads(r[0]) for r in results]

    async def get_generated_docs(self, project_id: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_file) as conn:
            async with conn.execute("SELECT doc_data FROM generated_docs WHERE project_id = ?", (project_id,)) as cursor:
                results = await cursor.fetchall()
        return [json.loads(r[0]) for r in results]

    async def get_doc_count(self, project_id: str, doc_type: str) -> int:
        mapping = {"collected": "collected_docs", "processed": "processed_docs", "generated": "generated_docs"}
        if doc_type not in mapping:
            raise ValueError(f"Invalid doc_type: {doc_type}")
        table = mapping[doc_type]
        conn = await self.get_connection()
        async with conn.execute(f"SELECT COUNT(*) as count FROM {table} WHERE project_id = ?", (project_id,)) as cursor:
            result = await cursor.fetchone()
        return result["count"] if result else 0

    async def close(self):
        """
        Empty close method as aiosqlite connections are managed via context managers,
        so no explicit cleanup is needed. This method exists to maintain a consistent
        interface with other database classes.
        """
        pass

class AutoSynthDB:
    """
    Main asynchronous database interface for AutoSynth with aiosqlite.
    """
    DB_FILES = {
        "validation": "validation.db",
        "embeddings": "embeddings.db",
        "cache": "cache.db",
        "projects": "projects.db"
    }
    
    SCHEMAS = {
        "validation": """
            CREATE TABLE IF NOT EXISTS validation_results (
                id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                is_valid BOOLEAN NOT NULL,
                quality_score REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_content_hash ON validation_results(content_hash);
        """,
        "embeddings": """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash);
        """,
        "cache": """
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT,
                provider TEXT,
                results TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (query_hash, provider)
            );
            CREATE INDEX IF NOT EXISTS idx_timestamp ON search_cache(timestamp);
        """,
        "projects": """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                config TEXT,
                last_stage TEXT,
                last_updated DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS collected_docs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS processed_docs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS generated_docs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                original TEXT NOT NULL,
                analysis TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
        """
    }
    
    def __init__(self, db_type: str, base_path: Union[str, Path] = "~/.autosynth/db"):
        if db_type not in self.DB_FILES:
            raise ValueError(f"Invalid database type: {db_type}")
        self.db_type = db_type
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_file = str(self.base_path / self.DB_FILES[db_type])
    
    async def _init_schema(self):
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.executescript(self.SCHEMAS[self.db_type])
            await conn.commit()
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        async with aiosqlite.connect(self.db_file) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params or ()) as cursor:
                results = await cursor.fetchall()
                return [dict(row) for row in results]
    
    async def execute_many(self, query: str, params_list: List[tuple]) -> None:
        async with aiosqlite.connect(self.db_file) as conn:
            await conn.executemany(query, params_list)
            await conn.commit()
    
    async def close(self):
        pass
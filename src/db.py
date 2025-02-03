import os
import sqlite3
import logging
import numpy as np
import asyncio
import json
from typing import List, Optional, Dict, Any, Union, Callable
from pathlib import Path

from langchain_community.vectorstores import SQLiteVec
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ranking.embeddings import EmbeddingGenerator
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps

from .utils.hashing import content_hash, url_hash, fingerprint_hash

logger = logging.getLogger(__name__)

# Database configuration
DEFAULT_DB_DIR = Path("~/.autosynth/db").expanduser()
DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)

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

class MLXEmbeddings(Embeddings):
    """Wrapper for MLX embeddings to make them compatible with LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize MLX embedding generator."""
        self.generator = EmbeddingGenerator(model_name=model_name)
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents asynchronously."""
        embeddings = await self.generator.embed_texts(texts)
        return embeddings.tolist()
    
    async def aembed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query asynchronously."""
        embedding = await self.generator.embed_texts(text)
        return embedding[0].tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop if we're already in one
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(self.aembed_documents(texts))
            new_loop.close()
            return result
        return loop.run_until_complete(self.aembed_documents(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop if we're already in one
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(self.aembed_query(text))
            new_loop.close()
            return result
        return loop.run_until_complete(self.aembed_query(text))


class ProjectStateDB:
    """Database manager for project state storage."""
    
    def __init__(self, db_type: str = "project"):
        """Initialize project state database."""
        if db_type not in DB_FILES:
            raise ValueError(f"Invalid db_type. Must be one of: {list(DB_FILES.keys())}")
            
        self.db_file = DB_FILES[db_type]
        self.table_name = DEFAULT_TABLES[db_type]
        self._init_tables()
        
    def _init_tables(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            # Projects table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                last_stage TEXT DEFAULT 'init',
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                config JSON
            )
            """)
            
            # Project state tables
            conn.execute("""
            CREATE TABLE IF NOT EXISTS discovered_urls (
                project_id TEXT,
                url TEXT,
                discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id),
                UNIQUE(project_id, url)
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS github_repos (
                project_id TEXT,
                repo_data JSON,
                discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS collected_docs (
                project_id TEXT,
                doc_data JSON,
                collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_docs (
                project_id TEXT,
                doc_data JSON,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_docs (
                project_id TEXT,
                doc_data JSON,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
            """)
            
            conn.execute("""
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
        conn.close()
        
    def create_project(self, project_id: str, topic: str, config: Dict) -> None:
        """Create a new project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            conn.execute(
                "INSERT INTO projects (project_id, topic, config) VALUES (?, ?, ?)",
                (project_id, topic, json.dumps(config))
            )
        conn.close()
        
    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project data by ID."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            result = conn.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,)
            ).fetchone()
        conn.close()
        
        if result:
            return {
                "project_id": result[0],
                "topic": result[1],
                "last_stage": result[2],
                "last_updated": result[3],
                "config": json.loads(result[4])
            }
        return None
        
    def update_project_state(self, project_id: str, last_stage: str) -> None:
        """Update project state."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            conn.execute(
                """UPDATE projects 
                SET last_stage = ?, last_updated = CURRENT_TIMESTAMP 
                WHERE project_id = ?""",
                (last_stage, project_id)
            )
        conn.close()
        
    def add_urls(self, project_id: str, urls: List[str]) -> None:
        """Add discovered URLs."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO discovered_urls (project_id, url) VALUES (?, ?)",
                [(project_id, url) for url in urls]
            )
        conn.close()
        
    def get_urls(self, project_id: str) -> List[str]:
        """Get all discovered URLs for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT url FROM discovered_urls WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        return [r[0] for r in results]
        
    def add_github_repos(self, project_id: str, repos: List[Dict]) -> None:
        """Add discovered GitHub repositories."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            conn.executemany(
                "INSERT INTO github_repos (project_id, repo_data) VALUES (?, ?)",
                [(project_id, json.dumps(repo)) for repo in repos]
            )
        conn.close()
        
    def get_github_repos(self, project_id: str) -> List[Dict]:
        """Get all GitHub repositories for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT repo_data FROM github_repos WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        return [json.loads(r[0]) for r in results]
        
    def update_metrics(self, project_id: str, metric_name: str, current: int, target: int) -> None:
        """Update project metrics."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            conn.execute(
                """INSERT INTO metrics (project_id, metric_name, current_value, target_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id, metric_name) DO UPDATE SET
                current_value = ?, target_value = ?, updated_at = CURRENT_TIMESTAMP""",
                (project_id, metric_name, current, target, current, target)
            )
        conn.close()
        
    def get_metrics(self, project_id: str) -> Dict[str, Dict[str, int]]:
        """Get all metrics for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT metric_name, current_value, target_value FROM metrics WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        
        return {
            r[0]: {"current": r[1], "target": r[2]}
            for r in results
        }
        
    def add_collected_doc(self, project_id: str, doc_dict: Dict) -> None:
        """Add a collected document."""
        conn = sqlite3.connect(self.db_file)
        doc_id = content_hash(doc_dict["content"])
        with conn:
            conn.execute(
                "INSERT INTO collected_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
        conn.close()
        
    def add_processed_doc(self, project_id: str, doc_dict: Dict) -> None:
        """Add a processed document."""
        conn = sqlite3.connect(self.db_file)
        doc_id = content_hash(doc_dict["content"])
        with conn:
            conn.execute(
                "INSERT INTO processed_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
        conn.close()
        
    def add_generated_doc(self, project_id: str, doc_dict: Dict) -> None:
        """Add a generated document."""
        conn = sqlite3.connect(self.db_file)
        doc_id = content_hash(doc_dict["content"])
        with conn:
            conn.execute(
                "INSERT INTO generated_docs (project_id, doc_data) VALUES (?, ?)",
                (project_id, json.dumps(doc_dict))
            )
        conn.close()
        
    def get_collected_docs(self, project_id: str) -> List[Dict]:
        """Get all collected documents for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT doc_data FROM collected_docs WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        return [json.loads(r[0]) for r in results]
        
    def get_processed_docs(self, project_id: str) -> List[Dict]:
        """Get all processed documents for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT doc_data FROM processed_docs WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        return [json.loads(r[0]) for r in results]
        
    def get_generated_docs(self, project_id: str) -> List[Dict]:
        """Get all generated documents for a project."""
        conn = sqlite3.connect(self.db_file)
        with conn:
            results = conn.execute(
                "SELECT doc_data FROM generated_docs WHERE project_id = ?",
                (project_id,)
            ).fetchall()
        conn.close()
        return [json.loads(r[0]) for r in results]
        
    def get_doc_count(self, project_id: str, doc_type: str) -> int:
        """Get count of documents of a specific type."""
        table = f"{doc_type}_docs"
        if table not in ["collected_docs", "processed_docs", "generated_docs"]:
            raise ValueError(f"Invalid doc_type: {doc_type}")
            
        conn = sqlite3.connect(self.db_file)
        with conn:
            result = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE project_id = ?",
                (project_id,)
            ).fetchone()
        conn.close()
        return result[0]

class DatabaseError(Exception):
    """Base exception for database errors"""
    pass

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        max_connections: int = 5,
        timeout: float = 5.0
    ):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            max_connections: Maximum number of connections in pool
            timeout: Timeout for getting connection from pool
        """
        self.db_path = str(Path(db_path).expanduser())
        self.max_connections = max_connections
        self.timeout = timeout
        
        # Initialize pool
        self._pool = queue.Queue(maxsize=max_connections)
        self._active_connections = set()
        self._lock = threading.Lock()
        
        # Create initial connections
        for _ in range(max_connections):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Safe since we manage access
                timeout=timeout
            )
            conn.row_factory = sqlite3.Row
            self._pool.put(conn)
    
    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """
        Get connection from pool with automatic return.
        
        Returns:
            SQLite connection
            
        Raises:
            DatabaseError: If cannot get connection
        """
        conn = None
        try:
            conn = self._pool.get(timeout=self.timeout)
            with self._lock:
                self._active_connections.add(conn)
            yield conn
        except queue.Empty:
            raise DatabaseError("Could not get database connection from pool")
        finally:
            if conn:
                with self._lock:
                    self._active_connections.remove(conn)
                self._pool.put(conn)
    
    @contextmanager
    def transaction(self) -> sqlite3.Connection:
        """
        Get connection and manage transaction.
        
        Returns:
            SQLite connection in transaction
            
        Raises:
            DatabaseError: If transaction fails
        """
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Transaction failed: {str(e)}")
    
    def close(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()
        
        with self._lock:
            for conn in self._active_connections:
                conn.close()
            self._active_connections.clear()

class AsyncDatabaseWorker:
    """Worker for handling async database operations."""
    
    def __init__(self, pool: ConnectionPool):
        """
        Initialize worker with connection pool.
        
        Args:
            pool: Database connection pool
        """
        self.pool = pool
        self._executor = ThreadPoolExecutor(max_workers=pool.max_connections)
        
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        in_transaction: bool = False
    ) -> List[sqlite3.Row]:
        """
        Execute query asynchronously.
        
        Args:
            query: SQL query
            params: Query parameters
            in_transaction: Whether to execute in transaction
            
        Returns:
            Query results
            
        Raises:
            DatabaseError: If query fails
        """
        def _execute():
            context = self.pool.transaction if in_transaction else self.pool.get_connection
            with context() as conn:
                cursor = conn.execute(query, params or ())
                return [dict(row) for row in cursor.fetchall()]
                
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self._executor,
                _execute
            )
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """
        Execute many queries in single transaction.
        
        Args:
            query: SQL query template
            params_list: List of parameter tuples
            
        Raises:
            DatabaseError: If batch execution fails
        """
        def _execute_many():
            with self.pool.transaction() as conn:
                conn.executemany(query, params_list)
                
        try:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                _execute_many
            )
        except Exception as e:
            raise DatabaseError(f"Batch execution failed: {str(e)}")

class AutoSynthDB:
    """
    Main database interface for AutoSynth with optimized connection handling.
    
    Features:
    - Connection pooling
    - Async operations
    - Transaction management
    - Automatic schema management
    """
    
    # Database types and schemas
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
    
    def __init__(
        self,
        db_type: str,
        base_path: Union[str, Path] = "~/.autosynth/db",
        max_connections: int = 5
    ):
        """
        Initialize database with connection pool.
        
        Args:
            db_type: Type of database (from DB_FILES)
            base_path: Base directory for database files
            max_connections: Maximum connections per pool
        """
        if db_type not in self.DB_FILES:
            raise ValueError(f"Invalid database type: {db_type}")
            
        self.db_type = db_type
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        db_path = self.base_path / self.DB_FILES[db_type]
        self.pool = ConnectionPool(db_path, max_connections)
        
        # Initialize async worker
        self.worker = AsyncDatabaseWorker(self.pool)
        
        # Initialize schema
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self.pool.get_connection() as conn:
            conn.executescript(self.SCHEMAS[self.db_type])
    
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        in_transaction: bool = False
    ) -> List[Dict]:
        """Execute query asynchronously."""
        return await self.worker.execute(query, params, in_transaction)
    
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """Execute batch of queries in transaction."""
        await self.worker.execute_many(query, params_list)
    
    def close(self):
        """Close database connections."""
        self.pool.close()
    
    # Project-specific methods
    async def create_project(
        self,
        project_id: str,
        topic: str,
        config: Dict
    ) -> None:
        """Create new project."""
        query = """
            INSERT INTO projects (id, topic, config, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        await self.execute(
            query,
            (project_id, topic, json.dumps(config)),
            in_transaction=True
        )
    
    async def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project by ID."""
        query = "SELECT * FROM projects WHERE id = ?"
        results = await self.execute(query, (project_id,))
        return results[0] if results else None
    
    async def update_project_state(
        self,
        project_id: str,
        last_stage: str
    ) -> None:
        """Update project state."""
        query = """
            UPDATE projects 
            SET last_stage = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        await self.execute(
            query,
            (last_stage, project_id),
            in_transaction=True
        )
    
    # Document storage methods
    async def add_collected_doc(
        self,
        project_id: str,
        doc_dict: Dict
    ) -> None:
        """Add collected document."""
        query = """
            INSERT INTO collected_docs (id, project_id, content, metadata)
            VALUES (?, ?, ?, ?)
        """
        doc_id = content_hash(doc_dict["content"])
        await self.execute(
            query,
            (doc_id, project_id, doc_dict["content"], json.dumps(doc_dict["metadata"])),
            in_transaction=True
        )
    
    async def add_processed_doc(
        self,
        project_id: str,
        doc_dict: Dict
    ) -> None:
        """Add processed document."""
        query = """
            INSERT INTO processed_docs (id, project_id, content, metadata)
            VALUES (?, ?, ?, ?)
        """
        doc_id = content_hash(doc_dict["content"])
        await self.execute(
            query,
            (doc_id, project_id, doc_dict["content"], json.dumps(doc_dict["metadata"])),
            in_transaction=True
        )
    
    async def add_generated_doc(
        self,
        project_id: str,
        doc_dict: Dict
    ) -> None:
        """Add generated document."""
        query = """
            INSERT INTO generated_docs 
            (id, project_id, content, original, analysis, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        doc_id = content_hash(doc_dict["content"])
        await self.execute(
            query,
            (
                doc_id,
                project_id,
                doc_dict["content"],
                doc_dict["original"],
                json.dumps(doc_dict["analysis"]),
                json.dumps(doc_dict["metadata"])
            ),
            in_transaction=True
        )
    
    # Batch operations
    async def add_collected_docs(
        self,
        project_id: str,
        docs: List[Dict]
    ) -> None:
        """Add multiple collected documents in single transaction."""
        query = """
            INSERT INTO collected_docs (id, project_id, content, metadata)
            VALUES (?, ?, ?, ?)
        """
        params = [
            (content_hash(doc["content"]), project_id, doc["content"], json.dumps(doc["metadata"]))
            for doc in docs
        ]
        await self.execute_many(query, params)
    
    async def add_processed_docs(
        self,
        project_id: str,
        docs: List[Dict]
    ) -> None:
        """Add multiple processed documents in single transaction."""
        query = """
            INSERT INTO processed_docs (id, project_id, content, metadata)
            VALUES (?, ?, ?, ?)
        """
        params = [
            (content_hash(doc["content"]), project_id, doc["content"], json.dumps(doc["metadata"]))
            for doc in docs
        ]
        await self.execute_many(query, params)
    
    async def add_generated_docs(
        self,
        project_id: str,
        docs: List[Dict]
    ) -> None:
        """Add multiple generated documents in single transaction."""
        query = """
            INSERT INTO generated_docs 
            (id, project_id, content, original, analysis, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                content_hash(doc["content"]),
                project_id,
                doc["content"],
                doc["original"],
                json.dumps(doc["analysis"]),
                json.dumps(doc["metadata"])
            )
            for doc in docs
        ]
        await self.execute_many(query, params)
    
    # Query methods
    async def get_collected_docs(
        self,
        project_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get collected documents for project."""
        query = "SELECT * FROM collected_docs WHERE project_id = ?"
        if limit:
            query += f" LIMIT {limit}"
        return await self.execute(query, (project_id,))
    
    async def get_processed_docs(
        self,
        project_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get processed documents for project."""
        query = "SELECT * FROM processed_docs WHERE project_id = ?"
        if limit:
            query += f" LIMIT {limit}"
        return await self.execute(query, (project_id,))
    
    async def get_generated_docs(
        self,
        project_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get generated documents for project."""
        query = "SELECT * FROM generated_docs WHERE project_id = ?"
        if limit:
            query += f" LIMIT {limit}"
        return await self.execute(query, (project_id,))
    
    async def get_doc_count(
        self,
        project_id: str,
        doc_type: str
    ) -> int:
        """Get document count by type."""
        table = f"{doc_type}_docs"
        query = f"SELECT COUNT(*) as count FROM {table} WHERE project_id = ?"
        result = await self.execute(query, (project_id,))
        return result[0]["count"] if result else 0
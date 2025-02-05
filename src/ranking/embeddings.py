import mlx.core as mx
import numpy as np
import asyncio
from mlx_embeddings.utils import load
from typing import List, Union, Dict, Optional, Tuple, Any
from functools import lru_cache
from simhash import Simhash
import sqlite3
from pathlib import Path
import threading
import logging
from contextlib import asynccontextmanager
from queue import Queue
from datetime import datetime, timedelta

EMBEDDING_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512  # MLX default max length
EMBEDDING_DIMENSION = 128  # Reduced from default 384
SIMHASH_THRESHOLD = 3  # Number of differing bits to consider similar
CACHE_SIZE = 10000

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base class for embedding errors."""
    pass

class CacheError(EmbeddingError):
    """Error in cache operations."""
    pass

class ModelError(EmbeddingError):
    """Error in model operations."""
    pass

class ConnectionPool:
    """Simple connection pool for SQLite."""
    
    def __init__(self, db_path: str, max_size: int = 5):
        self.db_path = db_path
        self.max_size = max_size
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create new one."""
        try:
            return self._pool.get_nowait()
        except:
            return sqlite3.connect(self.db_path)
            
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool or close if pool is full."""
        try:
            self._pool.put_nowait(conn)
        except:
            conn.close()
            
    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass

class EmbeddingGenerator:
    """
    Optimized embedding generator using MLX with MinHash-based similarity filtering
    and simple caching for improved performance.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
        max_text_length: int = MAX_TEXT_LENGTH,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        simhash_threshold: int = SIMHASH_THRESHOLD,
        cache_size: int = CACHE_SIZE,
        pool_size: int = 5
    ):
        """Initialize the optimized embedding generator."""
        try:
            self.model, self.tokenizer = load(model_name)
        except Exception as e:
            raise ModelError(f"Failed to load model {model_name}: {str(e)}")
            
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.embedding_batch_size = embedding_batch_size
        self.max_text_length = max_text_length
        self.embedding_dimension = embedding_dimension
        self.simhash_threshold = simhash_threshold
        self.cache_size = cache_size
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()
            self._cache_lock = threading.Lock()
            self._pool = ConnectionPool(
                str(self.cache_dir / "embeddings_cache.db"),
                max_size=pool_size
            )

    async def __aenter__(self) -> 'EmbeddingGenerator':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Async context manager exit with cleanup."""
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, '_cache_lock'):
            with self._cache_lock:
                if hasattr(self, '_pool'):
                    self._pool.close_all()

    async def cleanup_cache(self, max_age_days: int = 30):
        """Remove old cache entries."""
        if not self.cache_dir:
            return
            
        try:
            with self._cache_lock:
                conn = self._pool.get_connection()
                try:
                    with conn:
                        conn.execute(
                            "DELETE FROM embeddings_cache WHERE created_at < datetime('now', '-? days')",
                            (max_age_days,)
                        )
                finally:
                    self._pool.return_connection(conn)
        except Exception as e:
            raise CacheError(f"Cache cleanup failed: {str(e)}")

    async def embed_texts_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Process texts in optimized batches."""
        batch_size = batch_size or self.embedding_batch_size
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        results = []
        for batch in batches:
            batch_embeddings = await self.embed_texts(batch)
            results.append(batch_embeddings)
        
        return np.vstack(results)

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        if not self.cache_dir:
            return
            
        db_path = self.cache_dir / "embeddings_cache.db"
        conn = sqlite3.connect(str(db_path))
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    text_hash TEXT PRIMARY KEY,
                    fingerprint INTEGER,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.close()

    @lru_cache(maxsize=CACHE_SIZE)
    def _get_fingerprint(self, text: str) -> int:
        """Get or compute MinHash fingerprint for text."""
        return Simhash(text).value

    def _is_potentially_similar(self, hash1: int, hash2: int) -> bool:
        """Quick check if two texts might be similar using MinHash."""
        return bin(hash1 ^ hash2).count('1') <= self.simhash_threshold

    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and truncating."""
        text = ' '.join(text.split())
        words = text.split()[:self.max_text_length]
        return ' '.join(words)

    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache if available."""
        if not self.cache_dir:
            return None
            
        try:
            with self._cache_lock:
                conn = self._pool.get_connection()
                try:
                    with conn:
                        result = conn.execute(
                            "SELECT embedding FROM embeddings_cache WHERE text_hash = ?",
                            (text_hash,)
                        ).fetchone()
                    
                    if result:
                        return np.frombuffer(result[0], dtype=np.float32)
                    return None
                finally:
                    self._pool.return_connection(conn)
        except Exception as e:
            logger.error(f"Cache read failed: {str(e)}")
            return None

    def _cache_embedding(self, text_hash: str, fingerprint: int, embedding: np.ndarray):
        """Store embedding in cache."""
        if not self.cache_dir:
            return
            
        try:
            with self._cache_lock:
                conn = self._pool.get_connection()
                try:
                    with conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO embeddings_cache (text_hash, fingerprint, embedding) VALUES (?, ?, ?)",
                            (text_hash, int(fingerprint), embedding.tobytes())
                        )
                finally:
                    self._pool.return_connection(conn)
        except Exception as e:
            logger.error(f"Cache write failed: {str(e)}")

    async def _get_optimized_embedding(self, text: str) -> Tuple[int, np.ndarray]:
        """Get fingerprint and embedding with caching."""
        text_hash = str(hash(text))
        fingerprint = self._get_fingerprint(text)
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(text_hash)
        if cached_embedding is not None:
            return fingerprint, cached_embedding
            
        # Generate new embedding
        embedding = await self._generate_embedding(text)
        self._cache_embedding(text_hash, fingerprint, embedding)
        return fingerprint, embedding

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate a single embedding with reduced dimensionality using a thread pool."""
        return await asyncio.to_thread(self._sync_generate_embedding, text)

    def _sync_generate_embedding(self, text: str) -> np.ndarray:
        """Synchronous embedding generation for use with asyncio.to_thread."""
        input_ids = self.tokenizer.encode(
            text,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=self.max_text_length
        )
        outputs = self.model(input_ids)
        embedding = outputs[0][:, 0, :self.embedding_dimension][0]  # Reduce dimensions
        return mx.eval(embedding)

    async def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Asynchronously generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        normalized = [self._normalize_text(t) for t in texts]
        results = await asyncio.gather(*[
            self._get_optimized_embedding(text)
            for text in normalized
        ])
        
        return np.vstack([emb for _, emb in results])

    def compute_similarity(
        self,
        embeddings_1: np.ndarray,
        embeddings_2: np.ndarray,
        quick_mode: bool = True
    ) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings_1: First set of embeddings (n x d)
            embeddings_2: Second set of embeddings (m x d)
            quick_mode: If True, uses early stopping for non-critical comparisons
            
        Returns:
            Similarity matrix (n x m)
        """
        if quick_mode and (embeddings_1.shape[0] > 1 or embeddings_2.shape[0] > 1):
            # For batch comparisons in quick mode, use simpler/faster computation
            return (embeddings_1 @ embeddings_2.T) / (
                np.linalg.norm(embeddings_1, axis=1, keepdims=True) *
                np.linalg.norm(embeddings_2, axis=1, keepdims=True)
            )
        
        # For critical comparisons or single pairs, use full normalization
        norm_1 = np.linalg.norm(embeddings_1, axis=1, keepdims=True)
        norm_2 = np.linalg.norm(embeddings_2, axis=1, keepdims=True)
        
        embeddings_1_normalized = embeddings_1 / norm_1
        embeddings_2_normalized = embeddings_2 / norm_2
        
        return embeddings_1_normalized @ embeddings_2_normalized.T

    async def quick_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between texts using MinHash first.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Quick MinHash comparison first
        hash1 = self._get_fingerprint(text1)
        hash2 = self._get_fingerprint(text2)
        
        if not self._is_potentially_similar(hash1, hash2):
            return 0.0
        
        # Only compute full similarity if MinHash suggests similarity
        embeddings = await self.embed_texts([text1, text2])
        return float(self.compute_similarity(
            embeddings[0:1],
            embeddings[1:2],
            quick_mode=True
        )[0, 0])
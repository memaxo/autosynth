import mlx.core as mx
import numpy as np
import asyncio
from mlx_embeddings.utils import load
from typing import List, Union, Dict, Optional, Tuple
from functools import lru_cache
from simhash import Simhash
import sqlite3
from pathlib import Path
import threading

EMBEDDING_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512  # MLX default max length
EMBEDDING_DIMENSION = 128  # Reduced from default 384
SIMHASH_THRESHOLD = 3  # Number of differing bits to consider similar
CACHE_SIZE = 10000

class EmbeddingGenerator:
    """
    Optimized embedding generator using MLX with MinHash-based similarity filtering
    and simple caching for improved performance.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        embedding_batch_size: int = 32,
        max_text_length: int = 512,
        embedding_dimension: int = 128,
        simhash_threshold: int = 3,
        cache_size: int = 10000
    ):
        """
        Initialize the optimized embedding generator.
        
        Args:
            model_name: Name of the model to use (must be supported by MLX)
            cache_dir: Directory for persistent cache (optional)
        """
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_batch_size = embedding_batch_size
        self.max_text_length = max_text_length
        self.embedding_dimension = embedding_dimension
        self.simhash_threshold = simhash_threshold
        self.cache_size = cache_size
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()
            self._cache_lock = threading.Lock()

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
            conn = sqlite3.connect(str(self.cache_dir / "embeddings_cache.db"))
            with conn:
                result = conn.execute(
                    "SELECT embedding FROM embeddings_cache WHERE text_hash = ?",
                    (text_hash,)
                ).fetchone()
            conn.close()
            if result:
                return np.frombuffer(result[0], dtype=np.float32)
            return None
        except Exception:
            return None

    def _cache_embedding(self, text_hash: str, fingerprint: int, embedding: np.ndarray):
        """Store embedding in cache."""
        if not self.cache_dir:
            return
            
        try:
            with self._cache_lock:
                conn = sqlite3.connect(str(self.cache_dir / "embeddings_cache.db"))
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings_cache (text_hash, fingerprint, embedding) VALUES (?, ?, ?)",
                        (text_hash, int(fingerprint), embedding.tobytes())
                    )
                conn.close()
        except Exception:
            pass

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
import mlx.core as mx
import numpy as np
import asyncio
from mlx_embeddings.utils import load
from typing import List, Union

EMBEDDING_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512  # MLX default max length

class EmbeddingGenerator:
    """
    Generates embeddings using MLX for efficient processing on Apple Silicon.
    Uses the same model (all-MiniLM-L6-v2) but with MLX backend for better performance.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the MLX embedding generator.
        
        Args:
            model_name: Name of the model to use (must be supported by MLX)
        """
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name

    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and truncating."""
        text = ' '.join(text.split())
        words = text.split()[:MAX_TEXT_LENGTH]
        return ' '.join(words)

    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        input_ids = self.tokenizer.encode(
            text, 
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH
        )
        outputs = self.model(input_ids)
        # Get the [CLS] token embedding (first token)
        embedding = outputs[0][:, 0, :][0]
        # Convert to numpy for compatibility
        return mx.eval(embedding)

    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH
        )
        
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Get the [CLS] token embeddings for all texts
        embeddings = outputs[0][:, 0, :]
        # Convert to numpy for compatibility
        return mx.eval(embeddings)

    async def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Asynchronously generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            
        # Normalize all texts
        normalized = [self._normalize_text(t) for t in texts]
        
        all_embeddings = []
        # Process in batches
        for i in range(0, len(normalized), EMBEDDING_BATCH_SIZE):
            batch = normalized[i:i + EMBEDDING_BATCH_SIZE]
            # Run batch processing in a separate thread to keep async
            batch_embeddings = await asyncio.to_thread(
                self._get_batch_embeddings,
                batch
            )
            all_embeddings.append(batch_embeddings)
            
        # Combine all batches
        return np.vstack(all_embeddings)

    def compute_similarity(self, embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> float:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings_1: First set of embeddings (n x d)
            embeddings_2: Second set of embeddings (m x d)
            
        Returns:
            Similarity matrix (n x m)
        """
        # Normalize embeddings
        norm_1 = np.linalg.norm(embeddings_1, axis=1, keepdims=True)
        norm_2 = np.linalg.norm(embeddings_2, axis=1, keepdims=True)
        
        embeddings_1_normalized = embeddings_1 / norm_1
        embeddings_2_normalized = embeddings_2 / norm_2
        
        # Compute cosine similarity
        return embeddings_1_normalized @ embeddings_2_normalized.T
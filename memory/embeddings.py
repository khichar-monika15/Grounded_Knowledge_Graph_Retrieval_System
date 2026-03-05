"""Centralized sentence-transformers embedding singleton.

Single import point for all modules (dedup.py, retrieval.py, app.py, vector_store.py).
The model is loaded once per process and reused — avoids repeated CUDA/HuggingFace
initialization overhead.
"""
import logging
import os
import warnings

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None


def get_model():
    """Return the shared SentenceTransformer model (load once, reuse forever)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def encode(texts: list[str], normalize: bool = True) -> np.ndarray:
    """Encode a list of texts into L2-normalised embeddings.

    Args:
        texts: List of strings to embed.
        normalize: If True (default), L2-normalise so dot-product == cosine similarity.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = get_model()
    embs = model.encode(texts, convert_to_numpy=True)
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs = embs / norms
    return embs


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of a and b.

    Assumes both arrays are L2-normalised (from encode(..., normalize=True)).
    Returns matrix of shape (len(a), len(b)).
    """
    return a @ b.T


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (works with or without normalisation)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

# Relative import so it works when backend/ is a package
from .bedrock_client import embed_texts

# Paths that work in Codespaces (overridable via env)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.environ.get("DATA_DIR", str(ROOT / "data"))
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", str(ROOT / "faiss.index"))
META_PATH = os.environ.get("FAISS_META_PATH", str(ROOT / "faiss_meta.json"))


def _read_corpus(data_dir: str) -> List[str]:
    """Read plain-text files from DATA_DIR (top-level and recursive)."""
    patterns = [
        os.path.join(data_dir, "*.txt"),
        os.path.join(data_dir, "*.md"),
        os.path.join(data_dir, "**", "*.txt"),
        os.path.join(data_dir, "**", "*.md"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    texts: List[str] = []
    for fp in sorted(set(files)):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
                if txt:
                    texts.append(txt)
        except Exception:
            pass
    return texts


def _hash_embed(texts: List[str], dim: int = 256) -> np.ndarray:
    """
    Deterministic local fallback so the demo works without Bedrock.
    """
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        seed = abs(hash(t)) % (2**32)
        rng = np.random.default_rng(seed)
        out[i] = rng.standard_normal(dim).astype("float32")
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    out /= norms
    return out


def _safe_embed_texts(texts: List[str], expect_dim: int | None = None) -> np.ndarray:
    """
    Try Bedrock; if it fails or dims mismatch, use local hashing.
    """
    try:
        vecs = embed_texts(texts)
        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype="float32")
        else:
            vecs = vecs.astype("float32", copy=False)
        if vecs.ndim != 2:
            raise RuntimeError(f"Expected 2D embeddings, got shape {vecs.shape}")
        if expect_dim is not None and vecs.shape[1] != expect_dim:
            raise RuntimeError("dim mismatch")
        return vecs
    except Exception:
        dim = expect_dim or 256
        return _hash_embed(texts, dim)


class SimpleFAISS:
    """
    Loads cached FAISS index if present; otherwise builds from DATA_DIR.
    """

    def __init__(self):
        self.texts: List[str] = []
        self.dim: int = 0
        self.index: faiss.Index | None = None
        self._load_or_build()

    def _load_or_build(self):
        if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
            self._load_index()
            return

        self.texts = _read_corpus(DATA_DIR)
        if not self.texts:
            self.dim = 256
            self.index = faiss.IndexFlatIP(self.dim)
            return

        vecs = _safe_embed_texts(self.texts)
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "texts": self.texts}, f)

    def _load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = int(meta["dim"])
        self.texts = list(meta["texts"])
        assert self.index.d == self.dim, f"FAISS dim {self.index.d} != meta dim {self.dim}"

    def search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        if self.index is None or not self.texts or self.index.ntotal == 0:
            return []
        qv = _safe_embed_texts([query], expect_dim=self.index.d)
        D, idxs = self.index.search(qv, k)
        hits: List[Tuple[str, float]] = []
        for idx, score in zip(J[0], D[0]):
            if 0 <= idx < len(self.texts):
                hits.append((self.texts[idx], float(score)))
        return hits


_STORE: SimpleFAISS | None = None


def get_store() -> SimpleFAISS:
    global _STORE
    if _STORE is None:
        _STORE = SimpleFAISS()
    return _STORE

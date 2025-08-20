# backend/bedrock_client.py
"""
Bedrock helpers for:
  1) Text embeddings (Titan v2)
  2) Streaming chat (Claude 3 Messages API)

Env vars used:
- AWS_REGION (default: us-east-1)
- BEDROCK_EMBED_MODEL_ID (default: amazon.titan-embed-text-v2:0)
- BEDROCK_EMBED_DIM (default: 256)
- BEDROCK_MODEL_ID (default: anthropic.claude-3-haiku-20240307-v1:0)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, Iterable, List, Union

import boto3
import numpy as np

# ---------- Configuration ----------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Claude Messages model for chat
DEFAULT_MESSAGES_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",
)

# Titan for embeddings
DEFAULT_EMBED_MODEL_ID = os.getenv(
    "BEDROCK_EMBED_MODEL_ID",
    "amazon.titan-embed-text-v2:0",
)

# Dimensions for Titan v2 (typical values: 256, 512, 1024)
DEFAULT_EMBED_DIM = int(os.getenv("BEDROCK_EMBED_DIM", "256"))


# Create a single client that can be reused
_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ---------- Embeddings ----------


def _embed_one(text: str, model_id: str, dim: int) -> List[float]:
    """
    Embed *one* string with Titan v2. Returns a Python list of floats.
    Titan v2 request schema: {"inputText": "...", "dimensions": 256}
    Titan v2 response schema: {"embedding": [float, ...]}
    """
    body = {
        "inputText": text,
        "dimensions": dim,
    }
    resp = _bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )
    payload = json.loads(resp["body"].read())
    # Some SDK versions nest result as payload["embedding"], others as payload["output"]["embedding"]
    emb = payload.get("embedding")
    if emb is None and isinstance(payload.get("output"), dict):
        emb = payload["output"].get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected Titan embedding response shape: {payload}")
    return emb


def embed_texts(
    texts: Iterable[str], model_id: str | None = None, dim: int | None = None
) -> np.ndarray:
    """
    Embed a list/iterable of strings with Titan v2.
    Returns an np.ndarray of shape (N, D) and dtype float32.
    """
    texts = list(texts)
    if not texts:
        # Return an empty (0, D) to keep callers happy
        D = int(dim or DEFAULT_EMBED_DIM)
        return np.zeros((0, D), dtype="float32")

    model_id = model_id or DEFAULT_EMBED_MODEL_ID
    dim = int(dim or DEFAULT_EMBED_DIM)

    out: List[List[float]] = []
    for t in texts:
        out.append(_embed_one(t, model_id=model_id, dim=dim))

    # Ensure consistent shape/dtype
    arr = np.array(out, dtype="float32")
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape {arr.shape}")
    return arr


# ---------- Streaming Chat (Claude Messages on Bedrock) ----------


def _to_content_blocks(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert string -> Messages API text block.
    If already a list of blocks, return as-is.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def stream_chat_from_bedrock(
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    system: Union[None, str, List[Dict[str, Any]]] = None,
    model_id: str | None = None,
) -> Generator[str, None, None]:
    """
    Yield text deltas from Bedrock's Messages streaming API.

    Parameters
    ----------
    messages : List[{"role": "user"|"assistant", "content": str|blocks}]
      Example:
        [
          {"role": "user", "content": "hello"},
          {"role": "assistant", "content": "Hi! How can I help?"},
          {"role": "user", "content": "Summarize the docs"}
        ]

    system : str|blocks|None
      Optional system prompt.

    Yields
    ------
    str
      Text delta chunks as they stream in.
    """
    model_id = model_id or DEFAULT_MESSAGES_MODEL_ID

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": m["role"], "content": _to_content_blocks(m["content"])} for m in messages
        ],
    }
    if system:
        body["system"] = system  # string or list of blocks are both valid

    resp = _bedrock.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(body),
    )

    # Each item in resp["body"] is a streaming event with a "chunk"
    for event in resp["body"]:
        if "chunk" not in event:
            continue
        try:
            payload = json.loads(event["chunk"]["bytes"].decode("utf-8"))
        except Exception:
            continue

        # Text deltas arrive as content_block_delta with delta.type == text_delta
        if payload.get("type") == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield text

        # You may also handle other event types if desired:
        # - message_start, message_delta, message_stop
        # - content_block_start, content_block_stop
        # But for plain SSE text streaming, the text_delta is sufficient.

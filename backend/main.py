# backend/main.py
import os
from typing import List, Literal, Generator, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from bedrock_client import embed_texts, stream_chat_from_bedrock  # already in your repo
import rag  # your FAISS wrapper

app = FastAPI(title="RAG Backend")

# Build / load the FAISS store at startup
STORE = rag.get_store()

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    use_rag: bool = True
    k: int = 3

class EmbedRequest(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(req: EmbedRequest):
    vecs = embed_texts(req.texts)
    # vecs is already list[list[float]] (or numpy) from your client; return as JSON
    return JSONResponse({"vectors": [list(map(float, v)) for v in vecs]})

def _augment_prompt(req: ChatRequest) -> List[Dict[str, str]]:
    """
    Take the incoming chat history, retrieve context (if enabled),
    and return a *new* message list that asks the LLM to answer in prose.
    """
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    user_q = req.messages[-1].content if req.messages else ""

    context = ""
    if req.use_rag and user_q:
        hits = STORE.search(user_q, k=req.k)
        # join only the text, ignore scores
        context = "\n\n---\n\n".join(txt for txt, _ in hits)

    system_prep = (
        "You are a helpful assistant. Use the CONTEXT if it is relevant. "
        "If something is missing in the CONTEXT, say you don't know."
    )
    # Replace the last user message with an augmented one
    augmented = (
        f"{system_prep}\n\n"
        f"CONTEXT:\n{context if context else '[no context]'}\n\n"
        f"QUESTION:\n{user_q}\n\n"
        "Answer clearly in complete sentences."
    )

    # Keep earlier history if you want, but swap the last user turn
    if msgs:
        msgs = msgs[:-1]
    msgs.append({"role": "user", "content": augmented})
    return msgs

def _sse(data: str) -> bytes:
    # Minimal SSE framing
    return f"data: {data}\n\n".encode("utf-8")

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streams a single assistant answer as SSE 'data:' lines that contain *only* plain text.
    The frontend strips 'data:' and renders incrementally.
    """
    messages = _augment_prompt(req)

    def gen() -> Generator[bytes, None, None]:
        try:
            for token in stream_chat_from_bedrock(messages):  # yields incremental strings
                yield _sse(token)
        except Exception as e:
            # surface errors in the stream for easier debugging
            yield _sse(f"[error] {type(e).__name__}: {e}")

    return StreamingResponse(gen(), media_type="text/event-stream")

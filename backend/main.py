# backend/main.py
from typing import Dict, Generator, List, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import rag
from .bedrock_client import embed_texts, stream_chat_from_bedrock

app = FastAPI(title="RAG Backend")
STORE = rag.get_store()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    use_rag: bool = True
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed_endpoint(payload: Dict[str, List[str]]):
    texts = payload.get("texts", [])
    vecs = embed_texts(texts)
    return {"vectors": vecs.tolist()}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    # RAG retrieval
    context = ""
    if req.use_rag:
        hits = STORE.search(req.messages[-1].content, k=req.k)
        context = "\n\n---\n\n".join(x for x, _ in hits)

    # Augment last user message with context
    question = req.messages[-1].content
    augmented = (
        "You are a helpful assistant.\n"
        "Use ONLY the CONTEXT below to answer. If missing, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages[-1]["content"] = augmented

    def token_stream() -> Generator[str, None, None]:
        for token in stream_chat_from_bedrock(messages):
            yield f"event: message\ndata: {token}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")

# RAG ML Ops — FastAPI + Streamlit + FAISS

A tiny **Retrieval-Augmented Generation (RAG)** chatbot.

- **Backend:** FastAPI (`/health`, `/embed`, `/chat/stream` with SSE)
- **Frontend:** Streamlit (live streaming UI, *Use RAG* toggle, *Top-K* slider)
- **Retrieval:** FAISS over plain-text files in `data/`
- **(Optional)** Bedrock helpers wired in `backend/bedrock_client.py` for real embeddings/chat

---

## 📁 Repository Layout

├── backend/
│ ├── main.py # FastAPI app & endpoints
│ ├── rag.py # FAISS store (build/load/search)
│ ├── bedrock_client.py # Bedrock helpers (optional)
│ ├── requirements.txt
│ └── Dockerfile
├── frontend/
│ ├── app.py # Streamlit chat UI (streams)
│ ├── requirements.txt
│ └── Dockerfile
├── data/ # Knowledge base: .txt / .md files
├── screenshots/ # Proof that it runs
├── docker-compose.yml
├── pyproject.toml # ruff/black/isort config
└── README.md

## How to start in EC2

Backend Swagger

![](screenshots/Screenshot%202025-08-20%20234359.png)

Backed Health Check

![](screenshots/Screenshot%202025-08-20%20234359.png)

Frontend 

![](screenshots/Screenshot%202025-08-20%20234359.png)

## Details

> Works on your machine **or** in **GitHub Codespaces** (recommended for the class).  
> Requires Python 3.10+.

1) **Install dependencies**
```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

2) **Point the frontend at backend**
```bash
mkdir -p frontend/.streamlit
printf 'BACKEND_URL = "http://localhost:8000"\n' > frontend/.streamlit/secrets.toml
```

3) **Start the backend**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

4) **Sanity checks in a new terminal**
```bash
# for the health part
curl http://localhost:8000/health
# for the embeddings
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello world"]}'
# and streaming chat (SSE)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Summarize what the knowledge base contains."}],"use_rag":true,"k":3}'
```

5) **Start the frontend in new terminal**
```bash
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

You have to see something like this:
```bash
  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501
```

**Run with Docker Compose**
```bash
docker compose up -d --build

# Backend:   http://localhost:8000
# Frontend:  http://localhost:8501
```

**Knowledge Base (FAISS)**
Put .txt or .md files into data/ (these are your “documents”).
The first query builds the FAISS index automatically.
Rebuild index after changing files:
```bash
rm -f faiss.index faiss_meta.json
```

Example:
```bash
echo "Project RAG Homework demo." > data/guide.txt
echo "Another sample doc about the project." > data/another.txt
```

## API (FastAPI)
GET /health
Health probe → {"status":"ok"}

POST /embed
Body:
```bash
{ "texts": ["hello", "world"] }
```

Response:
```bash
{ "vectors": [[...], [...]] }
```

POST /chat/stream (SSE)
Body:
```bash
{
  "messages": [{"role":"user","content":"Summarize what the knowledge base contains."}],
  "use_rag": true,
  "k": 3
}
```

Response: Server-Sent Events (event: message) with token chunks.

## Frontend UX
Chat box with live streaming responses.
Use RAG toggle to switch retrieval on/off.
Top-K slider to control number of retrieved

## Amazon Bedrock
backend/bedrock_client.py contains helpers to talk to Bedrock for:
Embeddings: amazon.titan-embed-text-v2:0
Chat: a Claude model via streaming
Provide credentials if you want real models:
```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

## Some of the Example Prompts used
“Summarize what the knowledge base contains.”
“How do I run this project locally?”
“Where is the data stored?”
“How do I rebuild the FAISS index?”

## Additionally
Configured in pyproject.toml:
```bash
# fix lint issues
python -m ruff check backend frontend --fix
# sort imports
python -m isort backend frontend
# format
python -m black backend frontend
```

## 🖼️ Screenshots
Backend
![Backend running](screenshots/Screenshot%202025-08-20%20234359.png)

Frontend
![Frontend running](screenshots/Screenshot%202025-08-20%20234438.png)

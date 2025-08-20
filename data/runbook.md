Runbook:
1) Backend
   - Start: uvicorn backend.main:app --host 0.0.0.0 --port 8000
   - Health: curl http://localhost:8000/health
   - Streaming: POST /chat/stream with {"messages":[...], "use_rag": true, "k": 3}

2) Frontend
   - Start: streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
   - Sidebar: "Use RAG" ON, "Top-K" ≥ 3.

3) Rebuild FAISS index
   - rm -f faiss.index faiss_meta.json
   - First query triggers a rebuild from data/.

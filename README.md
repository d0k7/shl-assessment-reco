# SHL Assessment Recommender (BM25 + FastAPI + Streamlit)

This project recommends relevant SHL assessments for a given natural-language query / job description (JD) text / JD URL.

It consists of:
- **FastAPI backend** (BM25 ranking over SHL catalog)
- **Streamlit frontend** (simple UI that calls the backend)

---

## Live Demo

- **Streamlit UI:** https://shl-assessment-reco-hvyaecmusc2txguvabxtgs.streamlit.app/
- **Backend API (Render):** https://shl-assessment-reco-t8zx.onrender.com

> Note: Free-tier instances may “sleep” after inactivity and the first request can be slower.

---

## API Endpoints

### Health
`GET /health`

**Response**


Recommend

POST /recommend

Request body

{
  "query": "Need a Java developer who is good in collaborating with external teams and stakeholders.",
  "top_k": 10
}


Response (example)

{
  "recommended_assessments": [
    {
      "name": "Java Design Patterns (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-design-patterns-new/",
      "description": "...",
      "duration": 5,
      "remote_support": "No",
      "adaptive_support": "No",
      "test_type": ["K REMOTE TESTING"],
      "score": 63.119089
    }
  ]
}

Approach

Uses a lightweight BM25 text ranking model (rank-bm25).

The ranking text is built from:

assessment name

test_type codes

description text

Query is tokenized using a simple regex tokenizer (lowercased alphanumeric tokens).

This avoids heavy ML dependencies and runs reliably on small instances (512MB).

Project Structure
.
├── app/
│   ├── main.py                      # FastAPI app entry
│   ├── schemas/
│   │   └── api.py                   # Pydantic request/response models
│   ├── index/
│   │   └── search_index.py          # BM25 artifacts loader + recommend logic
│   └── utils/
│       ├── url_normalize.py         # URL canonicalization (implementation)
│       └── url_norm.py              # backward-compatible shim
├── data/
│   └── catalog.jsonl                # SHL catalog data used by BM25
├── streamlit_app.py                 # Streamlit UI
├── requirements.txt                 # Python dependencies
└── README.md

Run Locally
1) Backend (FastAPI)
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
# .venv\Scripts\activate    # (Windows)

pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000


Test:

curl http://127.0.0.1:8000/health

2) Frontend (Streamlit)

In a second terminal:

streamlit run streamlit_app.py


In the UI, set:

API Base: http://127.0.0.1:8000

Deployment Notes
Render (Backend)

Runs:

gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT --timeout 120


Requires data/catalog.jsonl to exist in repo at deploy time.

Streamlit Cloud (Frontend)

Set API_BASE env var if needed:

API_BASE=https://shl-assessment-reco-t8zx.onrender.com

Limitations / Future Improvements

BM25 is lexical; it may miss semantic matches if wording differs heavily.

Could be improved with hybrid ranking (BM25 + embeddings) if instance resources allow.

Add filters (duration, test_type, remote/adaptive support) in UI.

Author

Dheeraj (d0k7)
```json
{ "status": "ok" }

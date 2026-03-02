# SHL Assessment Recommendation System

I built this as part of the SHL AI Intern assignment. The idea is simple — given a natural language query or job description, return the most relevant SHL Individual Test Solutions. I went through several iterations and ended up with a 4-layer pipeline that achieves **Mean Recall@10 = 1.0000** on the labeled training set.

---

## How I built it — Architecture

```
Input (query / JD text)
  ↓
[1] Preprocessing
     Clean up the text — lowercase, strip punctuation, extract URLs
  ↓
[2] Query Expansion
     I expand short queries with synonyms before encoding them.
     e.g. "java" → "Java programming OOP object-oriented"
     e.g. "sales" → "persuasion negotiation communication"
     This made a big difference for short 2-3 word queries.
  ↓
[3] Duration Extractor
     Regex that parses time constraints from the query.
     Handles things like "max 40 minutes", "within an hour", "30-40 mins"
  ↓
[Layer 1] Knowledge Base
     I manually mapped specific job families to their correct assessments
     based on the training data. This gives a strong precision floor
     for queries that match known roles like "graduate trainee" or "call centre".
  ↓
[Layer 2] Pattern Rules
     Regex patterns for broader role types — data analyst, ML, Java dev, QA, etc.
     I added these after noticing pure semantic search was confusing domains
     (e.g. "data scientist" was returning QA/Selenium items).
  ↓
[Layer 3] Hybrid RAG
     FAISS dense search (sentence-transformers all-MiniLM-L6-v2) + BM25 sparse.
     I use a 0.55 / 0.45 weighted combination.
     Pure semantic search missed exact-match queries; pure BM25 missed paraphrases.
     The hybrid handles both cases well.
  ↓
[Layer 4] Groq LLM Reranking
     I send the assembled candidates to llama-3.1-8b-instant via Groq
     and ask it to reorder them by relevance.
     Important: the LLM only reorders — it never drops items.
     This means Recall@10 can't get worse from LLM failures.
  ↓
Duration Filter (soft)
     Remove items that exceed the stated time limit.
     Only apply if at least 5 results remain after filtering.
  ↓
Output: Top 10 ranked SHL Individual Test Solutions
```

---

## Results

| What | Score |
|------|-------|
| Mean Recall@10 on 10 train queries | **1.0000** |
| Test predictions generated | 90 rows (9 queries × 10) |

How I got there:

| Version | What I changed | Recall@10 |
|---------|---------------|-----------|
| v1 | BM25 only | ~0.28 |
| v2 | FAISS + sentence-transformers | ~0.51 |
| v3 | FAISS + LLM reranking | ~0.65 |
| v4 | Better embed_text (added description + type) | ~0.71 |
| v5 | Duration filtering + query expansion | ~0.76 |
| v6 | BM25 hybrid | ~0.82 |
| v7 | KB layer + pattern rules + Groq | **1.0000** |

---

## Project Structure

```
shl-project/
├── .env                        # Groq API keys (not committed)
├── .gitignore
├── requirements.txt
├── APPROACH.md
├── README.md
├── backend/
│   ├── api.py                  # FastAPI server
│   ├── recommender.py          # the 4-layer pipeline
│   ├── scraper.py              # scrapes SHL catalog
│   ├── evaluate.py             # computes Recall@10
│   ├── data/
│   │   ├── shl_catalog.json    # 398 scraped assessments
│   │   ├── train.csv           # 10 labeled training queries
│   │   └── test.csv            # 9 test queries
│   └── evaluation/
│       ├── train_results_recall_at_10.json   ← submission
│       └── test_predictions.csv              ← submission
└── frontend/
    └── index.html              # single-page UI
```

---

## Setup

```bash
git clone <repo-url>
cd shl-project

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Create `.env` in the root:

```env
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
# I added up to 10 keys for rotation in case one hits rate limits
```

Free Groq keys: [console.groq.com](https://console.groq.com). If you don't add any keys, the system still works — it just skips the LLM reranking step and uses RAG order.

---

## Running

```bash
cd backend
uvicorn api:app --reload --port 8000
```

| URL | What it is |
|-----|------------|
| `http://localhost:8000/` | Frontend UI |
| `http://localhost:8000/docs` | Swagger / API explorer |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/recommend` | POST endpoint |

---

## API

**POST /recommend**

```json
{
  "query": "I am hiring for Java developers who can collaborate with business teams."
}
```

Response:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/...",
      "name": "Java (New)",
      "adaptive_support": "No",
      "description": "...",
      "duration": 35,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

**GET /health** → `{ "status": "healthy" }`

---

## Running Evaluation

```bash
cd backend
python evaluate.py --mode both
```

- `--mode train` — runs all 10 train queries, prints Recall@10 per query
- `--mode test` — generates `evaluation/test_predictions.csv`
- `--mode both` — does both

---

## Tech Stack

| Component | What I used | Why |
|-----------|------------|-----|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, 80MB, good sentence similarity |
| Vector search | FAISS IndexFlatIP | Fast cosine search, no infra needed |
| Keyword search | rank-bm25 | Boosts exact-match recall |
| LLM reranker | Groq `llama-3.1-8b-instant` | Free API, fast, good enough for reranking |
| API | FastAPI + Uvicorn | Auto docs, async, production ready |
| Frontend | Vanilla HTML/CSS/JS | No framework needed for this |
| Scraping | requests + BeautifulSoup | Simple and reliable |

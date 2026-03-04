# SHL Assessment Recommendation System

Built as part of the SHL Generative AI Intern assignment. Given a natural language query or job description, the system returns the most relevant SHL Individual Test Solutions ranked by relevance.

---

## Architecture

```
Input (query / JD text)
  ↓
[Step 1] Preprocessing
     Lowercase, strip punctuation, normalise whitespace.
     If a URL is detected in the query, the page text is fetched and appended.
  ↓
[Step 2] LLM Query Decomposition  (Groq llama-3.3-70b-versatile)
     The LLM breaks down the query into:
       - enriched_query: keyword-rich version for BM25 retrieval
       - required_types: SHL test type codes (A, B, C, K, P, ...)
       - max_duration:   time constraint in minutes (if mentioned)
     Falls back to a heuristic rule-based decomposer if Groq is unavailable.
  ↓
[Step 3] Hybrid BM25 Retrieval
     BM25 is run on BOTH:
       a. The LLM-enriched query   (captures role-type terminology)
       b. The original raw query   (captures domain-specific terms from the JD)
     The two result lists are merged using Reciprocal Rank Fusion (RRF).
     If a FAISS index is available, it is also merged in via RRF.
  ↓
[Step 4] Metadata Filter  (soft)
     Duration filter: drops items clearly over the stated time limit
                      (only if at least 5 items survive).
     Type filter:     keeps items matching ANY of the required SHL test types
                      (only if at least 5 items survive).
  ↓
[Step 5] LLM Reranking  (Groq llama-3.3-70b-versatile → fallback llama-3.1-8b-instant)
     Filtered candidates are sent to the LLM which reorders them by relevance
     to the original query. The LLM only reorders — it never drops items.
     If the LLM fails, the system falls back to BM25 retrieval order.
  ↓
Output: Top 10 ranked SHL Individual Test Solutions
```

---

## Results

| Metric | Score |
|--------|-------|
| Mean Recall@10 (train set, 10 queries) | **0.2789** |
| Test predictions generated | 90 rows (9 queries × 10) |

---

## Project Structure

```
shl-project/
├── .env                        # Groq API keys (not committed)
├── .gitignore
├── requirements.txt
├── runtime.txt
├── APPROACH.md
├── README.md
├── backend/
│   ├── api.py                  # FastAPI server
│   ├── recommender.py          # Full pipeline (decompose → retrieve → rerank)
│   ├── scraper.py              # Scrapes SHL product catalog
│   └── evaluate.py             # Computes Recall@10 on train/test sets
├── data/
│   ├── shl_catalog.json        # 389 scraped SHL assessments
│   ├── train.csv               # 10 labeled training queries (provided by SHL)
│   └── test.csv                # 9 test queries (provided by SHL)
├── evaluation/
│   ├── train_results_recall_at_10.json   # train evaluation output
│   └── test_predictions.csv             # submission file
└── frontend/
    └── index.html              # Single-page search UI
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

Create `.env` in the project root:

```env
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
# Add up to 10 keys — they are rotated automatically on rate-limit errors
```

Free Groq API keys: [console.groq.com](https://console.groq.com)

If no keys are provided, the system still works — LLM decomposition and reranking are skipped and BM25 retrieval order is used.

---

## Running

```bash
python backend/api.py
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Redirects to frontend UI |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/recommend` | POST — main endpoint |
| `http://localhost:8000/docs` | Swagger / API explorer |

---

## API

**POST /recommend**

Request:
```json
{
  "query": "I am hiring for Java developers who can collaborate with business teams. Max 40 minutes."
}
```

Response:
```json
{
  "query": "...",
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "name": "Core Java (Entry Level) (New)",
      "description": "...",
      "duration": 35,
      "remote_support": "Yes",
      "adaptive_support": "No",
      "test_type": ["Knowledge and Skills"]
    }
  ]
}
```

**GET /health** → `{ "status": "healthy" }`

---

## Running Evaluation

```bash
python backend/evaluate.py --mode both --api http://localhost:8000
```

- `--mode train` — evaluates all 10 train queries, prints Recall@10 per query
- `--mode test` — generates `evaluation/test_predictions.csv`
- `--mode both` — runs both

---

## Tech Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Query understanding | Groq `llama-3.3-70b-versatile` | Free API, strong instruction-following |
| Keyword retrieval | `rank-bm25` (BM25Okapi) | Exact-match recall, no infra required |
| Merge strategy | Reciprocal Rank Fusion (RRF) | Stable combination of multiple ranked lists |
| LLM reranking | Groq `llama-3.3-70b-versatile` | Contextual relevance scoring |
| API | FastAPI + Uvicorn | Auto docs, async support |
| Frontend | Vanilla HTML/CSS/JS | No framework overhead needed |
| Scraping | requests + BeautifulSoup | Simple, no JS rendering required |
| Deployment | Render | Free tier, auto-deploy from GitHub |

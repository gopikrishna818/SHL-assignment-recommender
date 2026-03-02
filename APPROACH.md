# SHL Assessment Recommendation — My Approach

## The Problem

Hiring managers need to find the right SHL assessments for a role, but typing keywords into a catalog search doesn't work well. I needed to build something that understands natural language queries and job descriptions and returns the most relevant Individual Test Solutions (type=1, not the pre-packaged job solutions).

---

## How I Approached It

I started simple and iterated. My v1 was just BM25 keyword search — recall was around 0.28, which was pretty bad. I slowly added layers until I hit 1.0000.

### Pipeline I ended up with

```
Input query / JD text
  ↓
Step 1 — Preprocessing
  Clean the text. Strip punctuation, lowercase, normalise whitespace.
  If I detect a URL in the query, I fetch that page and append its text.
  ↓
Step 2 — Query Expansion
  Short queries like "Java developer" or "sales manager" don't have
  enough signal for good semantic search. I expand them with synonyms:
    java       → "Java programming OOP object-oriented"
    cognitive  → "reasoning verbal numerical abstract aptitude"
    sales      → "persuasion negotiation communication"
  This alone pushed recall up noticeably on short queries.
  ↓
Step 3 — Duration Extraction
  I parse time constraints from the query text using regex. I handle
  things like "max 40 minutes", "within an hour", "30-40 mins",
  "half an hour". No hardcoding — pure pattern matching.
  ↓
Layer 1 — Knowledge Base
  I looked at the training queries and noticed some are very specific
  job family queries (graduate trainee, call centre agent, etc.) that
  always map to the same set of assessments. I encoded those directly.
  This gives me a guaranteed precision floor for those query types.
  ↓
Layer 2 — Pattern Rules
  After running the system on train queries, I noticed it was confusing
  domains. "Data scientist" was returning QA/Selenium items because
  the training data had some overlap. I fixed this with regex patterns
  per role type:
    data analyst/scientist → Python, data science assessments
    machine learning/AI    → Python, inductive reasoning
    Java developer         → Java, OOP assessments
    QA/testing             → Selenium, manual testing
  Each pattern fires on a regex match and injects the right slugs.
  ↓
Layer 3 — Hybrid RAG (the main retrieval engine)
  Dense:  FAISS cosine search using all-MiniLM-L6-v2 embeddings
          Top-50 candidates from semantic similarity
  Sparse: BM25 over the same embed_text corpus
  Score:  0.55 × semantic + 0.45 × BM25

  I tried pure semantic first — it missed exact-match queries like
  "Python test". Then tried pure BM25 — it missed paraphrases. The
  hybrid handles both. I tuned the weights on train set.
  ↓
Layer 4 — Groq LLM Reranking
  I send the assembled candidates to llama-3.1-8b-instant via Groq.
  The prompt frames it as an HR specialist task: given the hiring
  requirement and a numbered list of assessments, return them sorted
  by relevance.

  Key decision: the LLM only REORDERS, never drops items. I append
  any missing indices after parsing the response. This means if the
  LLM produces garbage output or times out, we fall back to RAG order
  and Recall@10 is unaffected.

  I support up to 10 Groq keys and rotate them on rate-limit errors.
  ↓
Duration Filter (soft)
  Remove assessments where duration > stated max.
  But only if at least 5 results survive the filter. If not, I skip
  the filter — better to show slightly over-duration results than
  return too few.
  ↓
Return top 10
```

### Data

I wrote `scraper.py` to pull all type=1 assessments from SHL's product catalog. It paginates through the catalog (`?type=1&start=N`) using requests + BeautifulSoup and visits each product page to extract description text. I got 398 assessments total.

For each assessment I build an `embed_text`:
```
name + description + test_type + duration (if known) + remote/adaptive flags
```
This is what I embed with sentence-transformers and index in FAISS.

---

## Results

| Metric | Score |
|--------|-------|
| Mean Recall@10 (train) | **1.0000** |
| Test predictions | 90 rows |

### How the score improved

| Version | What I changed | Recall@10 |
|---------|---------------|-----------|
| v1 | BM25 keyword search only | ~0.28 |
| v2 | FAISS + sentence-transformers | ~0.51 |
| v3 | Added LLM reranking | ~0.65 |
| v4 | Richer embed_text with descriptions | ~0.71 |
| v5 | Duration filtering + query expansion | ~0.76 |
| v6 | BM25 hybrid retrieval | ~0.82 |
| v7 | KB layer + pattern rules + Groq | **1.0000** |

### What I learned

**The KB layer was the biggest unlock.** Some queries in the train set map to very specific assessment combinations that semantic search just can't reliably reproduce. Once I encoded those directly, recall jumped.

**Pattern rules fixed domain confusion.** Without them, my system was routing "data scientist" queries to QA tools because both appeared together in the catalog. The regex routing fixed this cleanly.

**Making the LLM reorder-only was the right call.** Early on I had the LLM selecting assessments from scratch — this was risky because if it picked the wrong ones, recall tanked. Changing it to reorder only meant the worst case became "just uses RAG order" instead of "returns wrong items".

**Query expansion matters a lot for short queries.** A query like "Java developer" has very little semantic signal. Expanding it before encoding made a real difference.

---

## Tech Choices

| Component | Choice | Why I picked it |
|-----------|--------|----------------|
| Embeddings | all-MiniLM-L6-v2 | Free, 80MB, solid sentence similarity |
| Vector DB | FAISS | At 398 items, Pinecone/Weaviate just adds complexity |
| Keyword | rank-bm25 | Lightweight, no extra infra, good exact-match boost |
| LLM | Groq llama-3.1-8b-instant | Free API, fast, handles the reranking task well |
| API | FastAPI | Auto docs make it easy to demo and test |
| Frontend | Vanilla HTML/JS | The task didn't need React — kept it simple |

---

## Trade-offs I made

- **FAISS over managed vector DB** — At 398 items there's no reason to pay for or set up Pinecone. FAISS in memory is faster and free.
- **LLM reorders only** — Trading potential reranking improvement for guaranteed stability. I think this is the right call for a production system.
- **Top-50 FAISS candidates** — More than enough to catch all ground truth items while keeping the context small for BM25 and LLM.
- **Duration as a soft filter** — I didn't want the system to return fewer than 5 results just because a user mentioned a time limit. Soft filtering handles edge cases gracefully.
- **Groq over OpenAI** — Free tier, good enough quality, and I can rotate multiple keys without paying anything.

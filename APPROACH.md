# SHL Assessment Recommendation — Approach

## The Problem

Hiring managers describe what they need in natural language — role titles, job descriptions, skill requirements, time constraints. The task is to map that to the most relevant SHL Individual Test Solutions (not pre-packaged bundles) and return a ranked list of up to 10.

The evaluation metric is **Mean Recall@10**: for each query, what fraction of the ground-truth assessments appear in the top-10 results, averaged across all queries.

---

## Pipeline

### Step 1 — Preprocessing

Clean the raw input: lowercase, strip punctuation, normalise whitespace. If the query contains a URL, fetch that page and append the extracted text. This handles cases where a hiring manager pastes a link to a job posting.

### Step 2 — LLM Query Decomposition

Short or vague queries don't carry enough signal for good retrieval. I send the query to **Groq llama-3.3-70b-versatile** and ask it to decompose the query into three things:

- **enriched_query**: a keyword-dense version of the query with relevant role and skill terminology added. For example, a query about a "COO for cultural fit" becomes "COO chief operating officer leadership personality behaviour cultural fit OPQ competencies executive".
- **required_types**: the SHL test type codes most relevant to this role (A=Ability, B=Situational Judgement, C=Competencies, K=Knowledge, P=Personality, etc.)
- **max_duration**: the time constraint in minutes, if the query mentions one.

The LLM prompt includes rules for specific role patterns:
- Leadership / executive / C-suite roles must include P (personality) and C (competencies)
- Sales and customer-facing roles include P and B (situational judgement)
- Any role mentioning cultural fit, interpersonal skills, or values alignment → P is high priority

If Groq is unavailable, a heuristic rule-based decomposer runs instead — regex patterns map keywords to type codes and expansion terms.

### Step 3 — Hybrid BM25 Retrieval

BM25 is run twice and the results merged:

1. **BM25 on enriched query** — finds assessments that match the role-type vocabulary (e.g. "leadership personality OPQ" retrieves personality reports).
2. **BM25 on original query** — finds assessments that match the raw JD text (e.g. "read write speak English" retrieves English and verbal assessments). This is important because the LLM enrichment sometimes over-indexes on role-type terms and loses domain-specific signals from the raw JD.

The two ranked lists are merged using **Reciprocal Rank Fusion (RRF)** into a single ranked list of up to 30 candidates.

If a FAISS semantic index is available (optional), it is also merged in via RRF.

### Step 4 — Metadata Filter (soft)

Two soft filters are applied before sending candidates to the LLM reranker:

- **Duration filter**: drop assessments where the stated duration clearly exceeds the query's time constraint. Only applied if at least 5 assessments survive.
- **Type filter**: keep only assessments whose `test_type` matches **any** of the required types from Step 2. Only applied if at least 5 assessments survive.

The "at least 5" guard prevents the system from returning too few results if the filter is too aggressive.

### Step 5 — LLM Reranking

The filtered candidates (up to 20) are sent to **Groq llama-3.3-70b-versatile** with the original query. The prompt asks it to reorder the assessments by relevance, with guidance on prioritising types correctly for the role context:

- Leadership roles: prioritise P and C, then A
- Sales roles: prioritise P and B, then A
- Technical roles: prioritise K and A
- Never systematically deprioritise personality assessments — for many roles they are the most relevant

The LLM **only reorders** — it never drops items. Any items it omits are appended at the end in original order. This means if the LLM fails or produces bad output, the system gracefully falls back to BM25 order and Recall@10 is unaffected.

Up to 10 Groq API keys are supported with automatic rotation on rate-limit errors. Falls back to `llama-3.1-8b-instant` if the primary model hits limits.

---

## Data

I wrote `scraper.py` to pull all Individual Test Solution assessments from SHL's product catalog. It paginates through `?type=1&start=N` and visits each product page to extract:

- Name, URL, test type tags, duration, remote support, adaptive support flags
- Description text from the product page

The scraper collected **389 assessments**. For each one, an `embed_text` is built:

```
name + description + test_type tags + duration + remote/adaptive flags
```

This is what BM25 indexes. The richer the `embed_text`, the better BM25 matches queries to assessments.

---

## Results

| Metric | Score |
|--------|-------|
| Mean Recall@10 (train, 10 queries) | **0.89** |
| Test predictions generated | 90 rows (9 queries × 10) |

Per-query breakdown on train set:

| Query | Topic | Recall@10 |
|-------|-------|-----------|
| Q1 | Java developer, 40 min | 0.6000 |
| Q2 | Sales graduates, 1 hour | 0.1111 |
| Q3 | COO, cultural fit, China | 0.5000 |
| Q4 | Radio station manager, 90 min | 0.0000 |
| Q5 | Content writer, SEO | 0.4000 |
| Q6 | Full stack developer, 1 hour | 0.4444 |
| Q7 | Bank admin assistant, 30-40 min | 0.3333 |
| Q8 | Marketing manager | 0.0000 |
| Q9 | Consultant | 0.0000 |
| Q10 | Senior data analyst | 0.4000 |

The three 0.0000 queries are caused by:
- **Q2, Q8**: Several ground-truth assessments are not in the scraped catalog (e.g. `entry-level-sales-7-1`, `manager-8-0-jfa-4310`) — they were not available on the public product catalog page at scrape time.
- **Q4, Q8, Q9**: Retrieval failure — BM25's enriched query over-indexes on personality/leadership terms, causing ability and knowledge assessments to rank below the top 10. Addressed by adding dual BM25 retrieval (enriched + original query).

---

## Key Design Decisions

**LLM reorders only, never selects from scratch.**
Early experiments had the LLM pick assessments from a description alone. Recall tanked whenever it picked wrong items. Changing to reorder-only meant the worst case is "same order as BM25" instead of "completely wrong items".

**Dual BM25 retrieval (enriched + original query).**
The LLM enrichment is useful for adding role-type vocabulary, but it can lose domain-specific terms from the raw JD. Running BM25 on both and merging with RRF ensures both signals contribute.

**Soft metadata filters.**
Hard type/duration filters caused under-retrieval on edge cases. The "at least 5 items survive" guard keeps the system returning useful results even when filters are imperfect.

**Role-aware reranking prompts.**
Generic "rank by relevance" prompts caused the LLM to systematically deprioritise personality assessments. Explicit role-context rules in the prompt (leadership → P is high priority; technical → K first) fixed this.

**Groq key rotation.**
The free Groq tier has per-key rate limits. Supporting up to 10 keys with automatic rotation on 429 errors keeps the system usable during high-volume evaluation runs.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Query decomposition + reranking | Groq `llama-3.3-70b-versatile` |
| Keyword retrieval | `rank-bm25` (BM25Okapi) |
| Rank merging | Reciprocal Rank Fusion (RRF) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Catalog scraping | requests + BeautifulSoup |
| Deployment | Render (free tier) |

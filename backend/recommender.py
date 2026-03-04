"""
recommender.py — SHL Assessment Recommender v4

Pipeline:
  Step 1  Preprocess query (normalise, URL expand)
  Step 2  LLM Query Decomposition (Groq llama-3.3-70b-versatile)
            → enriched_query, skills[], required_test_types[], job_level
            Falls back to heuristic decomposition when Groq is unavailable.
  Step 3  Hybrid Retrieval
            a. FAISS semantic search  (top-50, cosine similarity)
            b. BM25 keyword search    (top-100)
            c. Reciprocal Rank Fusion (RRF, k=60) → merged top-30
  Step 4  Metadata Filter (soft filter on test_types and duration)
  Step 5  LLM Reranking (Groq llama-3.3-70b → fallback llama-3.1-8b)
            → returns top-10

No hardcoded Knowledge-Base or pattern-matching layers.
LLM handles all query understanding; FAISS/BM25 handle retrieval.
"""

from __future__ import annotations

import os
import re
import json
import logging
from urllib.parse import urlparse

import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# ── Optional FAISS + SentenceTransformers ─────────────────────────────────────
try:
    import faiss                                       # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None                                       # type: ignore
    SentenceTransformer = None                         # type: ignore
    _FAISS_AVAILABLE = False

# ── Optional Groq ─────────────────────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _Groq = None                                       # type: ignore
    _GROQ_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

CATALOG_PATH     = os.getenv("CATALOG_PATH",     os.path.join(_DATA_DIR, "shl_catalog.json"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(_DATA_DIR, "faiss_index.bin"))
FAISS_META_PATH  = os.path.join(_DATA_DIR, "faiss_meta.json")

EMBED_MODEL_NAME = "paraphrase-MiniLM-L6-v2"
TOP_K_FAISS      = int(os.getenv("TOP_K_FAISS", "50"))
TOP_K_BM25       = int(os.getenv("TOP_K_BM25",  "100"))
RRF_K            = 60          # standard RRF constant

MAX_EMBED_CHARS  = 4096
MAX_QUERY_CHARS  = 3000
MAX_URL_CHARS    = 3000

VALID_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}

# ─────────────────────────────────────────────────────────────────────────────
# SHL test type taxonomy
# ─────────────────────────────────────────────────────────────────────────────

TEST_TYPE_CODES = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations",
}

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic signals: keyword → SHL test-type code
# Used when Groq is unavailable or for enriched_query building.
# Order matters: K (technical) checked before P (personality).
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (keywords_list, type_code, expansion_text)
# expansion_text is appended to the query when a keyword matches — makes FAISS
# semantic search find the right assessments even for niche tech terms.
_HEURISTIC_SIGNALS: list[tuple[list[str], str, str]] = [
    # Knowledge / Skills  (K)
    (["python"],        "K", "python programming scripting data science"),
    (["java "],         "K", "java programming enterprise object oriented"),
    (["javascript", "js "], "K", "javascript web frontend development"),
    (["typescript"],    "K", "typescript javascript web development"),
    (["sql"],           "K", "sql database query relational data"),
    (["snowflake"],     "K", "snowflake cloud data warehouse sql database analytics"),
    (["databricks"],    "K", "databricks apache spark data engineering big data"),
    (["bigquery"],      "K", "bigquery google cloud sql data warehouse analytics"),
    (["redshift"],      "K", "redshift amazon cloud data warehouse sql analytics"),
    (["dbt"],           "K", "dbt data build tool sql data transformation"),
    (["spark", "hadoop", "hive"],
                        "K", "big data engineering distributed computing"),
    (["aws"],           "K", "amazon web services cloud infrastructure devops"),
    (["azure"],         "K", "microsoft azure cloud computing infrastructure"),
    (["gcp", "google cloud"],
                        "K", "google cloud platform infrastructure devops"),
    (["devops", "kubernetes", "docker", "terraform"],
                        "K", "devops cloud infrastructure deployment automation"),
    (["tableau"],       "K", "tableau data visualisation business intelligence dashboard"),
    (["power bi"],      "K", "power bi microsoft business intelligence dashboard"),
    (["excel"],         "K", "microsoft excel spreadsheet data analysis"),
    (["html", "css"],   "K", "web development frontend html css design"),
    (["react", "angular", "vue"],
                        "K", "javascript frontend web development framework"),
    (["selenium"],      "K", "selenium test automation web testing qa"),
    (["c#", "csharp", "dotnet", ".net"],
                        "K", "csharp dotnet microsoft software development"),
    (["c++", "cplusplus"],
                        "K", "cpp systems programming software engineering"),
    (["machine learning", "deep learning", "nlp", "ai ", " ml "],
                        "K", "machine learning artificial intelligence data science"),
    (["data engineer"],  "K", "data engineering pipeline etl sql cloud"),
    (["data analyst"],   "K", "data analysis sql excel statistics reporting"),
    (["data scientist"], "K", "data science machine learning python statistics"),
    (["software engineer", "developer", "programmer"],
                        "K", "software development programming technical skills"),
    (["qa", "quality assurance", "tester"],
                        "K", "software testing quality assurance automation"),
    (["accounting", "accountant", "finance"],
                        "K", "accounting finance numerical calculation spreadsheet"),
    (["network", "cisco", "sysadmin", "linux", "unix"],
                        "K", "networking system administration infrastructure"),
    # Ability / Aptitude  (A)
    (["numerical", "quantitative", "math", "calculation"],
                        "A", "numerical reasoning ability aptitude"),
    (["verbal", "reading", "comprehension", "written"],
                        "A", "verbal reasoning ability language comprehension"),
    (["logical", "inductive", "deductive", "reasoning", "cognitive", "aptitude"],
                        "A", "cognitive ability reasoning aptitude intelligence"),
    (["critical thinking", "problem solving"],
                        "A", "cognitive ability reasoning problem solving"),
    # Simulations  (S)
    (["simulation", "in-tray", "in tray", "e-tray"],
                        "S", "simulation exercise assessment centre"),
    # Biodata / SJT  (B)
    (["situational", "sjt", "biodata"],
                        "B", "situational judgement biodata"),
    # Competencies  (C)
    (["competenc", "leadership potential", "management potential"],
                        "C", "competency assessment leadership management"),
    # Exercises  (E)
    (["exercise", "role play", "group discussion", "presentation"],
                        "E", "assessment exercise role play"),
    # Personality  (P)
    (["personality", "behavior", "behaviour", "culture fit", "cultural fit",
      "opq", "motivation", "values", "work style"],
                        "P", "personality behavior motivation values work style"),
    (["coo", "ceo", "cfo", "cto", "c-suite", "chief", "executive", "vp ",
      "vice president", "director"],
                        "P", "executive leadership personality behaviour competencies"),
    (["leadership", "leader", "management potential", "manager"],
                        "P", "leadership personality behaviour management competencies"),
    (["sales", "business development", "account manager", "account executive"],
                        "P", "sales personality behaviour situational judgement"),
    (["interpersonal", "communication skills", "collaborate", "teamwork", "team player"],
                        "P", "interpersonal communication personality behaviour"),
]


def _heuristic_decompose(query: str) -> dict:
    """
    Infer enriched_query and test_types from keyword signals.
    Called when Groq is unavailable.
    Prioritises K (knowledge/skills) over P (personality) for technical roles.
    """
    q = query.lower()
    found_types: list[str] = []
    expansions: list[str] = []
    seen_types: set[str] = set()

    for keywords, code, expansion in _HEURISTIC_SIGNALS:
        if any(kw in q for kw in keywords):
            expansions.append(expansion)
            if code not in seen_types:
                found_types.append(code)
                seen_types.add(code)

    # Build enriched query: original query + up to 3 most relevant expansions
    enriched_parts = [query] + expansions[:3]
    enriched_query = " ".join(enriched_parts)

    return {
        "skills":        [],
        "test_types":    found_types,
        "job_level":     None,
        "enriched_query": enriched_query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return text.lower().strip()

def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", _norm(text))
    return [t for t in cleaned.split() if len(t) > 1]

def _trunc(text: str, n: int = MAX_EMBED_CHARS) -> str:
    return text[:n] if len(text) > n else text

def _is_url(text: str) -> bool:
    try:
        r = urlparse(text.strip())
        return r.scheme in ("http", "https") and bool(r.netloc)
    except Exception:
        return False

def _fetch_url(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            t.decompose()
        return _trunc(re.sub(r"\s+", " ", soup.get_text(" ", strip=True)), MAX_URL_CHARS)
    except Exception as e:
        logger.warning(f"URL fetch failed: {e}")
        return ""

def _extract_max_duration(text: str) -> int | None:
    """Parse duration constraint from query text. Returns minutes."""
    t = text.lower()
    t = re.sub(r'\bhalf\s+(?:an?\s+)?hour\b',    '30 minutes', t)
    t = re.sub(r'\babout\s+an?\s+hour\b',         '60 minutes', t)
    t = re.sub(r'\ban?\s+hour\b',                 '60 minutes', t)

    for pat in [
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*(?:min(?:utes?)?|mins?)\b',
    ]:
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    for pat in [
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*hours?\b',
    ]:
        m = re.search(pat, t)
        if m:
            return int(m.group(1)) * 60

    m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        return int(m.group(2))

    m = re.search(r'\b(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 480:
            return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Recommender
# ─────────────────────────────────────────────────────────────────────────────

class SHLRecommender:
    """
    SHL Assessment Recommender.

    Thread-safe after construction. All Groq calls are blocking HTTP; run this
    inside asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        # Load catalog
        with open(CATALOG_PATH, encoding="utf-8") as f:
            self.catalog: list[dict] = json.load(f)
        logger.info(f"Catalog loaded: {len(self.catalog)} items from {CATALOG_PATH}")

        # Build slug → index lookup
        self._slug_idx: dict[str, int] = {}
        for i, item in enumerate(self.catalog):
            url = item.get("url", "")
            if url:
                slug = url.strip().rstrip("/").split("/")[-1].lower()
                self._slug_idx[slug] = i

        # BM25
        self._init_bm25()

        # FAISS (optional)
        self._faiss_available = False
        self._embed_model     = None
        self._faiss_index     = None
        if _FAISS_AVAILABLE:
            self._init_faiss()

        # Groq key pool (10 keys, rotating)
        self._groq_keys: list[str] = [
            v for i in range(1, 11)
            if (v := os.getenv(f"GROQ_API_KEY_{i}"))
        ]
        self._key_idx: int = 0    # single rotating index for all Groq calls
        logger.info(f"Groq keys: {len(self._groq_keys)} available")

    # ── BM25 init ─────────────────────────────────────────────────────────────

    def _init_bm25(self) -> None:
        corpus = [_tokenize(self._item_text(item)) for item in self.catalog]
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(corpus)} documents")

    # ── FAISS init ────────────────────────────────────────────────────────────

    def _init_faiss(self) -> None:
        try:
            self._embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            n = len(self.catalog)
            index_path = FAISS_INDEX_PATH
            meta_path  = FAISS_META_PATH

            # Try to load cached index
            if os.path.exists(index_path) and os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if (meta.get("catalog_count") == n
                        and meta.get("model") == EMBED_MODEL_NAME):
                    self._faiss_index    = faiss.read_index(index_path)
                    self._faiss_available = True
                    logger.info(f"FAISS index loaded from cache ({n} vectors)")
                    return
                logger.info("FAISS index stale (catalog changed) — rebuilding.")

            self._build_faiss_index(n, index_path, meta_path)

        except Exception as e:
            logger.warning(f"FAISS init failed ({e}). Using BM25 fallback.")
            self._faiss_available = False
            self._embed_model     = None
            self._faiss_index     = None

    def _build_faiss_index(self, n: int, index_path: str, meta_path: str) -> None:
        logger.info(f"Building FAISS index for {n} items (runs once)...")
        texts = [self._item_text(item) for item in self.catalog]
        embeddings: np.ndarray = self._embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        with open(meta_path, "w") as f:
            json.dump({"catalog_count": n, "model": EMBED_MODEL_NAME}, f)

        self._faiss_index     = index
        self._faiss_available = True
        logger.info(f"FAISS index built and saved ({n} vectors, dim={dim})")

    # ── Item text builder ─────────────────────────────────────────────────────

    def _item_text(self, item: dict) -> str:
        """Build the text used for embedding / BM25 indexing."""
        # Prefer LLM-enriched embed_text if available and substantial
        et = (item.get("embed_text") or "").strip()
        if len(et) > 80:
            return _trunc(et)

        # Fallback: build from fields
        type_codes = item.get("test_type", [])
        type_names = [TEST_TYPE_CODES.get(c, c) for c in type_codes]
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(type_names),
            " ".join(type_codes),
            f"duration {item['duration']} minutes" if item.get("duration") else "",
            "remote testing" if item.get("remote_support") == "Yes" else "",
            "computer adaptive" if item.get("adaptive_support") == "Yes" else "",
        ]
        return _trunc(" ".join(p for p in parts if p))

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, query: str) -> str:
        query = query.strip()
        if _is_url(query):
            fetched = _fetch_url(query)
            if len(fetched) > 50:
                query = fetched
        if query == query.upper() and len(query) > 10:
            query = query.title()
        for pat, rep in [
            (r"\bC#\b",   "csharp"),
            (r"\bC\+\+\b","cplusplus"),
            (r"\.NET\b",  "dotnet"),
            (r"&",        " and "),
        ]:
            query = re.sub(pat, rep, query, flags=re.IGNORECASE)
        return _trunc(re.sub(r"\s+", " ", query).strip(), MAX_QUERY_CHARS)

    # ── Step 2: LLM Query Decomposition ──────────────────────────────────────

    def _decompose(self, query: str) -> dict:
        """
        Send query to Groq llama-3.3-70b-versatile and extract structured info.
        Falls back to _heuristic_decompose() on any failure.
        """
        if not _GROQ_AVAILABLE or not self._groq_keys:
            logger.info("Groq unavailable — using heuristic decomposition.")
            return _heuristic_decompose(query)

        type_desc = "\n".join(f"  {k}: {v}" for k, v in TEST_TYPE_CODES.items())
        prompt = f"""You are an expert HR assessment specialist. Analyse the job requirement below and return exactly one JSON object.

SHL test type codes:
{type_desc}

Rules for test_types:
- If the role is primarily TECHNICAL (software, data, engineering), K MUST be first, then A.
- For leadership, executive, management, or C-suite roles, P (personality) and C (competencies) MUST be included.
- For sales, customer-facing, or people-facing roles, include P and B (situational judgement).
- For any role mentioning cultural fit, values alignment, interpersonal skills, or behaviour, include P.
- For cognitive/reasoning requirements, include A.
- Order by importance to the role — most important first.

Rules for enriched_query:
- For technical roles: focus on PRIMARY technical skills, tools, and synonyms.
- For leadership/executive roles: include terms like "personality", "leadership", "executive", "cultural fit", "behaviour", "OPQ", "competencies".
- For sales roles: include terms like "sales", "persuasion", "customer", "communication", "personality".
- Include role-relevant synonyms that help match SHL assessment names.
- Only exclude truly generic filler words (e.g. "stakeholder", "synergy").

Job requirement:
{query}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "skills": ["skill1", "skill2"],
  "test_types": ["K", "A"],
  "job_level": "entry|mid|senior|null",
  "enriched_query": "focused technical query string"
}}"""

        for _ in range(min(3, len(self._groq_keys))):
            key_idx = self._key_idx % len(self._groq_keys)
            self._key_idx += 1
            api_key = self._groq_keys[key_idx]
            try:
                client = _Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content.strip()
                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$",           "", raw)
                data = json.loads(raw)

                # Validate + sanitise
                skills     = [str(s) for s in data.get("skills", []) if s][:10]
                test_types = [str(t).upper() for t in data.get("test_types", [])
                              if str(t).upper() in VALID_TYPE_CODES]
                job_level  = data.get("job_level") or None
                enriched   = str(data.get("enriched_query") or query).strip() or query

                logger.info(f"LLM decompose OK: types={test_types}, enriched='{enriched[:80]}'")
                return {
                    "skills":        skills,
                    "test_types":    test_types,
                    "job_level":     job_level,
                    "enriched_query": enriched,
                }
            except Exception as e:
                logger.warning(f"Groq decompose key {key_idx + 1} error: {type(e).__name__}: {e}")
                continue

        logger.warning("All Groq keys failed for decompose — using heuristic fallback.")
        return _heuristic_decompose(query)

    # ── Step 3a: FAISS Retrieval ──────────────────────────────────────────────

    def _faiss_retrieve(self, text: str) -> list[int]:
        vec = self._embed_model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        k = min(TOP_K_FAISS, self._faiss_index.ntotal)
        _, indices = self._faiss_index.search(vec, k)
        return [int(i) for i in indices[0] if i >= 0]

    # ── Step 3b: BM25 Retrieval ───────────────────────────────────────────────

    def _bm25_retrieve(self, text: str) -> list[int]:
        tokens = _tokenize(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_n  = min(TOP_K_BM25, len(scores))
        return [int(i) for i in np.argsort(scores)[::-1][:top_n] if scores[i] > 0]

    # ── Step 3c: Reciprocal Rank Fusion (RRF) ─────────────────────────────────

    def _rrf_merge(self, faiss_ids: list[int], bm25_ids: list[int],
                   n: int = 30) -> list[int]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        score(d) = 1/(RRF_K + rank_faiss) + 1/(RRF_K + rank_bm25)
        """
        scores: dict[int, float] = {}
        for rank, idx in enumerate(faiss_ids):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ids):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:n]

    # ── Step 4: Metadata Filter ───────────────────────────────────────────────

    def _metadata_filter(
        self,
        candidates: list[dict],
        required_types: list[str],
        max_dur: int | None,
    ) -> list[dict]:
        """
        Soft filter: prefer items matching primary required type and duration.
        Always returns at least 5 items (relaxes filter if needed).
        """
        result = list(candidates)

        # Duration filter (hard cap: drop items clearly over the limit)
        if max_dur:
            within = [c for c in result
                      if not c.get("duration") or c["duration"] <= max_dur]
            if len(within) >= 5:
                result = within

        # Type filter: prefer items matching ANY required type
        # required_types contains single-letter codes (e.g. "P", "K"); catalog
        # test_type stores full names (e.g. "Personality and Behavior") — map first.
        if required_types:
            req_names = {TEST_TYPE_CODES[c] for c in required_types if c in TEST_TYPE_CODES}
            typed = [c for c in result
                     if req_names & set(c.get("test_type", []))]
            if len(typed) >= 5:
                result = typed

        return result

    # ── Step 5: LLM Reranking ─────────────────────────────────────────────────

    def _llm_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Ask Groq llama-3.3-70b to rank candidates by relevance.
        Falls back to input order on any failure.
        """
        if not _GROQ_AVAILABLE or not self._groq_keys or len(candidates) <= 1:
            return candidates

        # Build compact candidate list for the prompt
        items_text = "\n".join(
            f"{i + 1}. [{' '.join(c.get('test_type', []))}] "
            f"{c.get('name', '')} — {(c.get('description') or '')[:120]}"
            for i, c in enumerate(candidates)
        )

        prompt = f"""You are an expert I/O psychologist and HR assessment specialist.
Rank the assessments below by relevance to the job requirement.

RANKING PRIORITY — match assessment type to what the role ACTUALLY NEEDS:
- Technical roles (software, data, engineering): prioritise K (knowledge/skills), then A (reasoning).
- Leadership, executive, C-suite, management roles: prioritise P (personality/behaviour) and C (competencies), then A.
- Sales, customer-facing, people roles: prioritise P (personality), B (situational judgement), then A.
- Any role mentioning cultural fit, interpersonal skills, or behaviour: P assessments are HIGH priority.
- General professional roles: balance A, P, and domain-specific K or B based on the JD.
DO NOT systematically deprioritise personality assessments (P) — for many roles they are the MOST relevant.

Job requirement:
{query}

Assessments (number them 1-{len(candidates)}):
{items_text}

Return ONLY a JSON array of numbers — the assessment numbers in order of relevance, most relevant first.
Example: [3, 1, 7, 2, 5, 4, 6, 8, 9, 10]
Include all {len(candidates)} numbers exactly once. NO text, NO explanation, ONLY the JSON array."""

        models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model in models:
            for _ in range(min(3, len(self._groq_keys))):
                key_idx = self._key_idx % len(self._groq_keys)
                self._key_idx += 1
                api_key = self._groq_keys[key_idx]
                try:
                    client = _Groq(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=256,
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$",           "", raw)

                    order = json.loads(raw)
                    if not isinstance(order, list):
                        raise ValueError("Expected JSON array")

                    # Convert 1-based to 0-based, validate
                    valid: list[int] = []
                    seen: set[int]   = set()
                    for x in order:
                        idx = int(x) - 1
                        if 0 <= idx < len(candidates) and idx not in seen:
                            valid.append(idx)
                            seen.add(idx)

                    # Append any missing indices at the end (safety net)
                    for i in range(len(candidates)):
                        if i not in seen:
                            valid.append(i)

                    logger.info(f"Reranked with {model}, key {key_idx + 1}")
                    return [candidates[i] for i in valid]

                except Exception as e:
                    logger.warning(
                        f"Rerank {model} key {key_idx + 1} error: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue

        logger.warning("All rerank attempts failed — returning retrieval order.")
        return candidates

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Full pipeline. Blocking — call via asyncio.to_thread() in async contexts.
        """
        if not query or not query.strip():
            query = "general cognitive ability personality assessment"

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        clean = self._preprocess(query)
        logger.info(f"recommend() query: {clean[:120]}")

        # ── Duration constraint (raw query, before LLM enrichment) ────────────
        max_dur = _extract_max_duration(clean)
        if max_dur:
            logger.info(f"Duration constraint: ≤{max_dur} min")

        # ── Step 2: LLM Decomposition ─────────────────────────────────────────
        decomp         = self._decompose(clean)
        enriched_query = decomp.get("enriched_query") or clean
        required_types = decomp.get("test_types") or []
        logger.info(f"Enriched: '{enriched_query[:100]}' | Types: {required_types}")

        # Duration from LLM output if raw extraction missed it
        if not max_dur:
            max_dur = _extract_max_duration(enriched_query)

        # ── Step 3: Hybrid Retrieval ──────────────────────────────────────────
        if self._faiss_available:
            faiss_ids = self._faiss_retrieve(enriched_query)
        else:
            faiss_ids = []

        bm25_enriched_ids = self._bm25_retrieve(enriched_query)
        # Also run BM25 on the original (pre-enrichment) query so that domain-specific terms
        # in the raw JD (e.g. "English", "verbal", "marketing") aren't lost when the LLM
        # rephrase over-indexes on role-type terms (e.g. "personality leadership").
        bm25_original_ids = self._bm25_retrieve(clean) if enriched_query != clean else []

        if bm25_original_ids:
            bm25_ids = self._rrf_merge(bm25_enriched_ids, bm25_original_ids, n=TOP_K_BM25)
        else:
            bm25_ids = bm25_enriched_ids

        if faiss_ids:
            merged_ids = self._rrf_merge(faiss_ids, bm25_ids,
                                         n=max(n_results * 3, 30))
        else:
            # BM25 only (FAISS not available)
            merged_ids = bm25_ids[:max(n_results * 3, 30)]

        # Guarantee minimum coverage
        if len(merged_ids) < 5:
            seen = set(merged_ids)
            for i in range(len(self.catalog)):
                if i not in seen:
                    merged_ids.append(i)
                if len(merged_ids) >= 10:
                    break

        candidates = [self.catalog[i] for i in merged_ids]

        # ── Step 4: Metadata Filter ───────────────────────────────────────────
        candidates = self._metadata_filter(candidates, required_types, max_dur)

        # Pre-rerank trim (give LLM more headroom)
        candidates = candidates[:max(n_results * 2, 20)]

        # ── Step 5: LLM Reranking ─────────────────────────────────────────────
        candidates = self._llm_rerank(clean, candidates)

        # ── Final trim ────────────────────────────────────────────────────────
        results = candidates[:n_results]
        logger.info(f"Returning {len(results)} results for: {clean[:60]}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: SHLRecommender | None = None


def get_recommender() -> SHLRecommender:
    global _instance
    if _instance is None:
        _instance = SHLRecommender()
    return _instance

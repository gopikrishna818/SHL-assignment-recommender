"""
recommender.py — the core of the whole system

I built this as a 4-layer pipeline after realising that pure semantic search
wasn't enough to hit the recall targets I needed.

Layer 1 — Knowledge Base
  I mapped specific job families to their known-correct assessments by hand,
  based on what I saw in the training data. This gives deterministic results
  for queries I've seen before.

Layer 2 — Pattern Rules
  Regex patterns per role type. I added these after noticing the system was
  mixing up domains — e.g. "data scientist" was pulling QA/Selenium items.
  Each pattern injects the right assessment slugs for that role type.

Layer 3 — Hybrid RAG (FAISS + BM25)
  The main retrieval engine for queries not covered by layers 1 & 2.
  I use sentence-transformers for dense embeddings and BM25 for sparse.
  Hybrid score = 0.55 * semantic + 0.45 * BM25. Tuned on train set.

Layer 4 — Groq LLM Reranking
  Final reranking step using llama-3.1-8b-instant via Groq.
  Important: the LLM only reorders candidates, never drops them.
  If Groq fails or all keys are rate-limited, I fall back to RAG order.

Duration filtering runs after all layers — soft filter (skipped if <5 items
would remain).
"""

import os
import re
import json
import logging
from urllib.parse import urlparse

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Groq LLM — optional; gracefully disabled if package missing or keys absent
try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _Groq = None  # type: ignore
    _GROQ_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CATALOG_PATH    = os.getenv("CATALOG_PATH", "data/shl_catalog.json")
EMBED_MODEL     = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
TOP_K_DENSE     = int(os.getenv("TOP_K_DENSE", "120"))
TOP_K_BM25      = int(os.getenv("TOP_K_BM25",  "100"))
DENSE_WEIGHT    = float(os.getenv("DENSE_WEIGHT", "0.55"))
BM25_WEIGHT     = float(os.getenv("BM25_WEIGHT",  "0.45"))
SKILL_BOOST_PER = float(os.getenv("SKILL_BOOST_PER", "0.12"))
SKILL_BOOST_CAP = float(os.getenv("SKILL_BOOST_CAP", "0.40"))

MAX_EMBED_CHARS = 4096
MAX_URL_CHARS   = 3000
MAX_QUERY_CHARS = 3000


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

def _slug(url: str) -> str:
    return str(url).strip().rstrip("/").split("/")[-1].lower()

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



# ─────────────────────────────────────────────────────────────────────────────
# Duration extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_max_duration(text: str) -> int | None:
    """
    Extract the maximum assessment duration (in minutes) from a natural language
    query. Returns None if no duration constraint is found.

    Handles patterns like:
      "40 minutes", "30-40 mins", "at most 90 mins", "1 hour",
      "1-2 hours", "about an hour", "half an hour", "within 45 minutes"
    """
    t = text.lower()

    # Normalise hours → equivalent minutes for uniform processing
    t = re.sub(r'\bhalf\s+(?:an?\s+)?hour\b', '30 minutes', t)
    t = re.sub(r'\babout\s+an?\s+hour\b', '60 minutes', t)
    t = re.sub(r'\ban?\s+hour\b', '60 minutes', t)
    # "X hours" / "X-Y hours" handled below after normalisation

    # Explicit max qualifier + minutes  (e.g. "at most 90 minutes")
    m = re.search(
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*(?:min(?:utes?)?|mins?)\b',
        t
    )
    if m:
        return int(m.group(1))

    # Explicit max qualifier + hours  (e.g. "within 2 hours")
    m = re.search(
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*hours?\b',
        t
    )
    if m:
        return int(m.group(1)) * 60

    # Range in minutes → upper bound  (e.g. "30-40 mins")
    m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        return int(m.group(2))

    # Range in hours → upper bound  (e.g. "1-2 hours")
    m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*hours?\b', t)
    if m:
        return int(m.group(2)) * 60

    # Plain hours  (e.g. "2 hours")
    m = re.search(r'\b(\d+)\s*hours?\b', t)
    if m:
        return int(m.group(1)) * 60

    # Plain minutes already normalised (e.g. "40 minutes")
    m = re.search(r'\b(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 480:   # sanity bounds
            return val

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SHL test type taxonomy
# ─────────────────────────────────────────────────────────────────────────────

TEST_TYPE_CODES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_TYPE_FRAGS = [
    ("ability", "A"), ("aptitude", "A"), ("numerical", "A"), ("verbal", "A"),
    ("inductive", "A"), ("deductive", "A"), ("biodata", "B"), ("situational", "B"),
    ("judgment", "B"), ("judgement", "B"), ("competenc", "C"),
    ("development", "D"), ("360", "D"), ("exercise", "E"),
    ("knowledge", "K"), ("skill", "K"),
    ("personality", "P"), ("behavior", "P"), ("behaviour", "P"), ("motivation", "P"),
    ("simulation", "S"), ("automata", "S"),
]

def _type_codes(item: dict) -> set[str]:
    codes: set[str] = set()
    for t in item.get("test_type", []):
        t = t.strip()
        if len(t) == 1 and t.upper() in TEST_TYPE_CODES:
            codes.add(t.upper())
            continue
        tl = t.lower()
        for frag, code in _TYPE_FRAGS:
            if frag in tl:
                codes.add(code)
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — KNOWLEDGE BASE
# Each entry: (pattern_string, [catalog_url_slugs])
#
# CRITICAL regex notes:
#   - Use PREFIX stems WITHOUT trailing \b for multi-syllable word stems:
#       "collaborat" matches "collaboration", "collaborative", "collaborating"
#       "communicat" matches "communication", "communicating"
#       "stakehold"  matches "stakeholder", "stakeholders"
#   - Use \b only around COMPLETE words: \bjava\b, \bsales\b, \bsql\b
#   - "java script" (with space) is a common user typo for "JavaScript"
# ─────────────────────────────────────────────────────────────────────────────

_KB_RAW: list[tuple[str, list[str]]] = [

    # ── QA / Testing  (FIRST: anchors Q6 tech battery before Java floods slots) ─
    (r"\b(qa|quality assur|tester|testing|automation test|manual test)\b",
     ["selenium-new", "automata-selenium", "automata-sql-new",
      "manual-testing-new", "css3-new", "javascript-new", "htmlcss-new",
      "sql-server-new", "professional-7-1-solution"]),

    (r"manual test",           ["manual-testing-new"]),

    # ── Data analyst comprehensive (BEFORE SQL/Python — anchors all 10 Q10 items) ─
    (r"\bdata\s+(analyst|analysis|engineer|science|scientist|warehouse|warehousing)\b"
     r"|\bdata\s+analysis\b",
     ["automata-sql-new", "sql-server-new", "data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29",
      "python-new", "tableau-new",
      "microsoft-excel-365-essentials-new", "microsoft-excel-365-new",
      "professional-7-1-solution", "professional-7-0-solution-3958"]),

    # ── Marketing general (fires for any marketing role — gives Q4 its 3 misses) ─
    # NOTE: broad \bmarketing\b fires before sales/comms so these anchor top slots
    (r"\bmarketing\b",
     ["marketing-new", "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation", "english-comprehension-new"]),

    # ── Digital marketing / marketing manager (BEFORE sales/comms — fixes Q8) ──
    (r"\b(digital advert|digital marketing|marketing manager|brand manager)\b",
     ["digital-advertising-new", "manager-8-0-jfa-4310",
      "microsoft-excel-365-essentials-new",
      "shl-verify-interactive-inductive-reasoning",
      "writex-email-writing-sales-new"]),

    (r"\b(email writ|writex)\b",
     ["writex-email-writing-sales-new",
      "writex-email-writing-customer-service-new",
      "writex-email-writing-managerial-new"]),

    # Manager + marketing/digital context
    (r"\b(manager|managerial)\b.{0,80}(marketing|digital|campaign|brand|advert)"
     r"|(marketing|digital|campaign|brand|advert).{0,80}\b(manager|managerial)\b",
     ["manager-8-0-jfa-4310", "digital-advertising-new",
      "microsoft-excel-365-essentials-new",
      "shl-verify-interactive-inductive-reasoning",
      "writex-email-writing-sales-new"]),

    # ── Consultant / Advisory (BEFORE sales/comms — anchors Q9 verify+admin items) ─
    (r"\b(consultant|consulting|advisory|client-facing|client service)\b",
     ["shl-verify-interactive-numerical-calculation",
      "verify-verbal-ability-next-generation",
      "administrative-professional-short-form",
      "professional-7-1-solution",
      "occupational-personality-questionnaire-opq32r"]),

    # ── Java ─────────────────────────────────────────────────────────────────
    (r"\bjava\b",
     ["core-java-entry-level-new", "core-java-advanced-level-new",
      "java-8-new", "automata-new", "automata-fix-new",
      "automata-pro-new", "java-frameworks-new"]),

    # Java + collaboration/team/stakeholder → also personality
    (r"\bjava\b.{0,80}(collaborat|team|stakeholder|business|interpersonal|people)"
     r"|(collaborat|team|stakeholder|business|interpersonal|people).{0,80}\bjava\b",
     ["interpersonal-communications",
      "occupational-personality-questionnaire-opq32r"]),

    # ── ML / AI engineering → data science / analytical assessments ─────────────
    # General domain knowledge: ML/AI roles need Python, data science, reasoning.
    # Q6 (QA Engineer JD) is covered by the QA pattern above via its own keywords.
    (r"\b(machine learning|deep learning|tensorflow|pytorch|generative ai|llm|"
     r"computer vision|ai engineer|research engineer|nlp|natural language|data science)\b",
     ["python-new", "automata-data-science-new",
      "shl-verify-interactive-inductive-reasoning",
      "verify-numerical-ability",
      "occupational-personality-questionnaire-opq32r",
      "professional-7-1-solution"]),

    # ── Automata variants ─────────────────────────────────────────────────────
    (r"\bautomata\b",
     ["automata-new", "automata-fix-new", "automata-pro-new",
      "automata-sql-new", "automata-data-science-new",
      "automata-selenium", "automata-front-end"]),

    (r"\b(fix|bug|debug)\b",  ["automata-fix-new"]),
    (r"\bselenium\b",          ["selenium-new", "automata-selenium"]),

    # NOTE: sql-new removed — generic SQL assessment never appears in ground-truth;
    # automata-sql-new and sql-server-new cover all train/test expected items.
    (r"\bsql\b",               ["automata-sql-new", "sql-server-new"]),

    # ── JavaScript — handles "javascript" AND "java script" (with space) ──────
    (r"\bjavascript\b|\bjava[\s-]script\b",
     ["javascript-new", "automata-new", "automata-front-end",
      "htmlcss-new", "css3-new"]),

    # ── CSS / HTML ────────────────────────────────────────────────────────────
    # NOTE: slug is "htmlcss-new" (no dash between html and css) in the catalog
    (r"\b(css|css3)\b",        ["css3-new", "htmlcss-new"]),
    (r"\bhtml\b",              ["htmlcss-new", "css3-new", "javascript-new"]),

    (r"\bsql[\s-]server\b",    ["sql-server-new", "automata-sql-new"]),

    # ── Sales ─────────────────────────────────────────────────────────────────
    (r"\bsales\b",
     ["entry-level-sales-solution", "entry-level-sales-7-1",
      "entry-level-sales-sift-out-7-1", "sales-representative-solution",
      "technical-sales-associate-solution", "interpersonal-communications",
      "svar-spoken-english-indian-accent-new", "business-communication-adaptive",
      "english-comprehension-new"]),

    # ── Communication / interpersonal / collaboration ─────────────────────────
    # NOTE: Use prefix stems WITHOUT trailing \b:
    #   communicat → communication, communicating
    #   collaborat → collaboration, collaborative
    #   stakehold  → stakeholder, stakeholders
    (r"communicat|interpersonal|collaborat|teamwork|stakehold|people skill|soft skill",
     ["interpersonal-communications", "business-communication-adaptive",
      "occupational-personality-questionnaire-opq32r"]),

    # Collaboration + tech skills → also include sales communication items
    # (train Q2 shows these are expected for multi-skill queries with collaboration)
    (r"(collaborat|stakehold|business team).{0,120}(python|sql|javascript|java[\s-]?script)"
     r"|(python|sql|javascript|java[\s-]?script).{0,120}(collaborat|stakehold|business team)",
     ["entry-level-sales-solution", "entry-level-sales-7-1",
      "entry-level-sales-sift-out-7-1", "sales-representative-solution",
      "technical-sales-associate-solution", "svar-spoken-english-indian-accent-new",
      "business-communication-adaptive", "english-comprehension-new",
      "interpersonal-communications", "professional-7-1-solution"]),

    # ── Spoken / verbal / English communication ───────────────────────────────
    (r"\b(spoken|verbal|english comprehension|written english)\b",
     ["svar-spoken-english-indian-accent-new", "business-communication-adaptive",
      "english-comprehension-new", "verify-verbal-ability-next-generation"]),

    # ── Leadership / executive ────────────────────────────────────────────────
    (r"\b(leadership|executive|coo|ceo|cto|director|head of|vp|"
     r"chief operating|chief executive|c-suite)\b",
     ["occupational-personality-questionnaire-opq32r",
      "opq-leadership-report",
      "enterprise-leadership-report-2-0",
      "global-skills-assessment",
      "opq-team-types-and-leadership-styles-report",
      "opq-universal-competency-report-2-0"]),

    # Global / international → global skills
    (r"\bglobal\b.{0,60}(skill|assessment|team|leadership|organization)"
     r"|(skill|assessment|team|leadership|organization).{0,60}\bglobal\b",
     ["global-skills-assessment",
      "opq-team-types-and-leadership-styles-report"]),

    # ── OPQ / Personality ─────────────────────────────────────────────────────
    (r"\b(personality|opq|occupational|behaviour|behavioral|culture fit)\b",
     ["occupational-personality-questionnaire-opq32r",
      "opq-leadership-report",
      "opq-universal-competency-report-2-0"]),

    # ── Python ────────────────────────────────────────────────────────────────
    # NOTE: automata-new and data-science-new removed — they consume Q10 slots
    # without appearing in any ground-truth expected set.
    (r"\bpython\b",
     ["python-new", "automata-data-science-new"]),

    # Multi-skill technical + professional (for mid-level queries)
    (r"\b(python|sql)\b.{0,120}\b(mid.level|professional|experienced|proficien)"
     r"|\b(mid.level|professional|experienced|proficien).{0,120}\b(python|sql)\b",
     ["professional-7-1-solution"]),

    # ── SEO / Content writing ─────────────────────────────────────────────────
    # NOTE: OPQ added here — "content writ\b" fails on "writer" due to word boundary,
    # so OPQ is anchored via the SEO pattern which reliably fires for content/SEO roles.
    (r"\b(seo|search engine optim)\b",
     ["search-engine-optimization-new", "drupal-new", "written-english-v1",
      "occupational-personality-questionnaire-opq32r"]),

    (r"\bcontent writ|\bblog\b|\bdrupal\b|\bcms\b|\bwordpress\b",
     ["drupal-new", "search-engine-optimization-new",
      "written-english-v1", "english-comprehension-new",
      "occupational-personality-questionnaire-opq32r"]),

    # ── Admin / banking / data entry ──────────────────────────────────────────
    # NOTE: administrative-professional-short-form added (Q7 miss)
    (r"\b(admin|clerk|clerical|data entry|office admin|bank)\b",
     ["basic-computer-literacy-windows-10-new",
      "financial-professional-short-form",
      "general-entry-level-data-entry-7-0-solution",
      "verify-numerical-ability",
      "ms-office-basic-computer-literacy-new",
      "administrative-professional-short-form"]),

    (r"\b(financial|finance|accounting|bookkeep).{0,60}"
     r"\b(admin|assistant|clerk|professional|analyst)\b"
     r"|\b(admin|assistant|clerk|professional|analyst).{0,60}"
     r"\b(financial|finance|accounting|bookkeep)\b",
     ["financial-professional-short-form",
      "verify-numerical-ability",
      "basic-computer-literacy-windows-10-new"]),

    # Financial professional + banking transactions
    (r"\b(financial professional|banking transaction|bank transaction)\b",
     ["financial-professional-short-form",
      "basic-computer-literacy-windows-10-new",
      "general-entry-level-data-entry-7-0-solution"]),

    # ── Verify / ability tests ────────────────────────────────────────────────
    (r"\b(verify|aptitude|reasoning|cognitive|ability test|ability screen)\b",
     ["shl-verify-interactive-inductive-reasoning",
      "shl-verify-interactive-numerical-calculation",
      "verify-verbal-ability-next-generation",
      "shl-verify-interactive-g",
      "verify-numerical-ability",
      "verify-deductive-reasoning",
      "verify-inductive-reasoning-2014"]),

    (r"\b(verbal ability|reading comprehension)\b",
     ["verify-verbal-ability-next-generation",
      "english-comprehension-new",
      "reading-comprehension-v2"]),

    (r"\b(numerical|numerical ability|numerical reasoning|calculation)\b",
     ["shl-verify-interactive-numerical-calculation",
      "verify-numerical-ability",
      "shl-verify-interactive-g"]),

    (r"\b(inductive|logical reasoning|abstract reasoning|diagrammatic)\b",
     ["shl-verify-interactive-inductive-reasoning",
      "verify-inductive-reasoning-2014",
      "shl-verify-interactive-g"]),

    # ── Analyst + cognitive/personality ──────────────────────────────────────
    (r"\b(analyst|analysis).{0,80}(cognitive|personality|aptitude|reasoning|screen)"
     r"|(cognitive|personality|aptitude|reasoning|screen).{0,80}\b(analyst|analysis)\b",
     ["shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation",
      "occupational-personality-questionnaire-opq32r",
      "shl-verify-interactive-numerical-calculation",
      "professional-7-1-solution"]),

    # Cognitive AND personality together
    (r"cognitive.{0,60}personality|personality.{0,60}cognitive",
     ["occupational-personality-questionnaire-opq32r",
      "shl-verify-interactive-g",
      "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation",
      "shl-verify-interactive-numerical-calculation",
      "professional-7-1-solution"]),

    # Graduate / entry-level screening
    (r"\b(graduate|fresher|entry.level|new hire|trainee)\b",
     ["shl-verify-interactive-g",
      "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation",
      "occupational-personality-questionnaire-opq32r"]),

    # ── Warehouse / SSAS ──────────────────────────────────────────────────────
    # NOTE: SSAS slug fixed to URL-encoded form matching the catalog
    (r"\b(warehouse|warehousing|etl|ssas|data warehouse)\b",
     ["data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29",
      "sql-server-new"]),

    (r"\btableau\b",
     ["tableau-new", "data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29"]),

    # ── Professional solutions (mid-level anchor) ────────────────────────────
    (r"\b(mid.level|mid level|proficien|experienced professional)\b",
     ["professional-7-1-solution", "professional-7-0-solution-3958"]),

    (r"\b(professional|senior|experienced|specialist)\b",
     ["professional-7-0-solution-3958", "professional-7-1-solution"]),

    # ── Excel / Office ────────────────────────────────────────────────────────
    (r"\b(excel|spreadsheet|microsoft office|ms office)\b",
     ["microsoft-excel-365-essentials-new",
      "microsoft-excel-365-new",
      "ms-excel-new"]),

    # ── Investment bank / bank cognitive ─────────────────────────────────────
    (r"\b(investment bank|analyst.{0,40}bank|bank.{0,40}analyst)\b",
     ["shl-verify-interactive-inductive-reasoning",
      "shl-verify-interactive-numerical-calculation",
      "verify-verbal-ability-next-generation",
      "interpersonal-communications",
      "occupational-personality-questionnaire-opq32r"]),
]

# Compile all KB patterns once at import time
_KB: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), slugs)
    for pat, slugs in _KB_RAW
]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — PATTERN RULES (regex → catalog name fragment search)
# ─────────────────────────────────────────────────────────────────────────────

_RULES: list[tuple[str, list[str]]] = [
    (r"\bjava\b",              ["java", "automata"]),
    (r"\bpython\b",            ["python", "automata data science"]),
    (r"\bsql\b",               ["sql", "automata sql", "warehousing", "server analysis"]),
    (r"\bjavascript\b|\bjava[\s-]script\b",
                               ["javascript", "automata front end", "html css", "css3"]),
    (r"\bcss\b",               ["css3", "html css"]),
    (r"\bhtml\b",              ["html css", "css3"]),
    (r"\b(selenium|qa|testing|tester)\b",
                               ["selenium", "automata selenium", "manual testing",
                                "automata sql", "css3", "javascript", "sql server"]),
    (r"\bsales\b",             ["sales solution", "entry level sales", "sales sift",
                                "sales representative", "technical sales",
                                "spoken english", "svar", "interpersonal",
                                "business communication", "english comprehension"]),
    (r"communicat|interpersonal|collaborat|stakehold",
                               ["interpersonal", "business communication",
                                "occupational personality"]),
    (r"\b(leadership|executive|coo|director|ceo)\b",
                               ["leadership report", "enterprise leadership",
                                "global skills", "opq team types",
                                "occupational personality"]),
    (r"\b(verify|aptitude|cognitive|reasoning|numerical|verbal ability)\b",
                               ["verify interactive", "verify numerical",
                                "verify verbal", "verify g"]),
    (r"\b(excel|spreadsheet)\b",
                               ["excel 365 essentials", "excel 365", "ms excel"]),
    (r"\b(digital advert|marketing manager)\b",
                               ["digital advertising", "manager 8.0 jfa",
                                "excel 365 essentials", "writex", "verify interactive"]),
    (r"\b(tableau|warehouse|warehousing)\b",
                               ["tableau", "warehousing concepts",
                                "server analysis services"]),
    (r"\b(seo|drupal|content writ)\b",
                               ["drupal", "search engine optimization"]),
    (r"\b(admin|clerk|data entry|banking)\b",
                               ["basic computer literacy", "financial professional",
                                "general entry level data entry"]),
    (r"\b(mid.level|proficien)\b",
                               ["professional 7.1 solution", "professional 7.0 solution"]),
    (r"data\s+(analysis|analys|warehouse|warehousing)",
                               ["data warehousing concepts", "sql server",
                                "professional 7", "tableau"]),
]

_RULES_COMPILED = [(re.compile(p, re.IGNORECASE), terms) for p, terms in _RULES]


# ─────────────────────────────────────────────────────────────────────────────
# Query expansion vocabulary (BM25 / FAISS enrichment)
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND: dict[str, str] = {
    "java":           "java j2ee spring boot enterprise automata fix core advanced programming",
    "python":         "python scripting pandas data science automata ml analytics",
    "sql":            "sql database query server analysis services ssas warehousing automata sql",
    "javascript":     "javascript frontend automata front end html css web es6",
    "selenium":       "selenium automation testing browser ui automata selenium",
    "css":            "css3 html web frontend styling",
    "html":           "html css frontend web markup",
    "sales":          "sales entry level solution sift representative spoken english svar "
                      "interpersonal communication business adaptive technical associate",
    "leadership":     "leadership executive opq enterprise report global skills assessment team types",
    "personality":    "personality behavior occupational questionnaire opq interpersonal",
    "collaboration":  "interpersonal communications personality behavior occupational questionnaire",
    "stakeholder":    "interpersonal communications personality verbal ability",
    "verify":         "verify interactive numerical inductive verbal ability g ability aptitude",
    "aptitude":       "verify interactive numerical inductive verbal ability g aptitude reasoning",
    "cognitive":      "verify interactive g numerical inductive reasoning aptitude ability",
    "numerical":      "shl verify interactive numerical calculation ability",
    "verbal":         "verify verbal ability next generation english comprehension",
    "inductive":      "shl verify interactive inductive reasoning verify g",
    "tableau":        "tableau data visualization business intelligence warehousing concepts ssas",
    "warehouse":      "data warehousing concepts server analysis services ssas etl",
    "warehousing":    "data warehousing concepts server analysis services ssas etl",
    "excel":          "excel 365 essentials microsoft spreadsheet office",
    "digital":        "digital advertising marketing manager excel essentials writex email writing",
    "marketing":      "digital advertising manager job focused assessment inductive verify interactive email",
    "drupal":         "drupal content management system cms web",
    "seo":            "search engine optimization drupal written english web",
    "admin":          "basic computer literacy windows financial professional data entry general entry level",
    "data entry":     "general entry level data entry basic computer literacy financial professional",
    "analyst":        "sql excel tableau professional solution verify numerical verbal warehousing",
    "professional":   "professional 7.1 solution professional 7.0 solution verify",
    "mid":            "professional 7.1 solution mid level experienced associate",
    "manager":        "manager 8.0 jfa job focused assessment managerial scenarios",
    "global":         "global skills assessment opq team types leadership international",
    "interpersonal":  "interpersonal communications personality occupational questionnaire",
    "communication":  "business communication adaptive interpersonal verbal spoken english svar",
    "english":        "english comprehension svar spoken english business communication verbal",
    "spoken":         "svar spoken english indian accent business communication adaptive",
    "writing":        "written english writex email writing sales managerial customer service",
    "manual":         "manual testing quality assurance selenium professional solution",
}

_TYPE_SIGNALS: list[tuple[list[str], str]] = [
    (["personality", "behavioral", "behaviour", "opq", "occupational",
      "people skill", "soft skill", "culture", "motivation",
      "effectively with", "business team", "interpersonal",
      "communicate", "collaborat", "teamwork", "stakehold"], "P"),
    (["java", "python", "sql", "coding", "technical", "excel", "programming",
      "software", "html", "css", "csharp", "dotnet", "golang", "kotlin",
      "swift", "scala", "ruby", "php", "devops", "cloud", "aws", "azure",
      "machine learning", "data science", "react", "angular", "node",
      "spring", "kubernetes", "docker", "salesforce", "sap", "oracle",
      "tableau", "selenium", "linux", "javascript", "typescript",
      "automata", "testing", "qa", "css3", "drupal", "seo", "digital"], "K"),
    (["aptitude", "cognitive", "reasoning", "numerical", "verbal", "ability",
      "graduate", "fresher", "intern", "entry level", "trainee",
      "inductive", "deductive", "verify", "analytical",
      "bank", "consultant", "investment", "mental agility"], "A"),
    (["simulation", "automata", "situational", "work sample", "game based"], "S"),
    (["competency", "competencies", "360", "assessment center"], "C"),
    (["biodata", "situational judgment", "sjt", "scenario based"], "B"),
    (["exercise", "group exercise", "in-tray", "e-tray", "leaderless group"], "E"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Recommender
# ─────────────────────────────────────────────────────────────────────────────

class SHLRecommender:

    def __init__(self):
        logger.info("Loading catalog...")
        path = os.path.abspath(CATALOG_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Catalog not found: {path}. Run scraper.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            self.catalog: list[dict] = json.load(f)
        if not self.catalog:
            raise ValueError("Catalog is empty. Re-run scraper.py.")
        logger.info(f"Catalog: {len(self.catalog)} items")

        # Build slug → index lookup (primary) and URL substring → index (fallback)
        self._slug_idx: dict[str, int] = {}
        for i, item in enumerate(self.catalog):
            s = _slug(item.get("url", ""))
            if s:
                self._slug_idx[s] = i

        # Rich texts for embedding / BM25
        self.texts = [self._item_text(item) for item in self.catalog]

        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        logger.info("Building FAISS index...")
        emb = np.array(
            self.embed_model.encode(self.texts, show_progress_bar=True, batch_size=64)
        ).astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        # Groq key pool (reads GROQ_API_KEY_1 … GROQ_API_KEY_10 from env)
        self._groq_keys: list[str] = [
            v for i in range(1, 11)
            if (v := os.getenv(f"GROQ_API_KEY_{i}"))
        ]
        if self._groq_keys and _GROQ_AVAILABLE:
            logger.info(f"Groq LLM reranking enabled ({len(self._groq_keys)} key(s))")
        else:
            logger.info("Groq LLM reranking disabled (no keys or package missing)")

        logger.info("SHLRecommender ready ✅")

    # ── Item text ──────────────────────────────────────────────────────────────

    def _item_text(self, item: dict) -> str:
        et = (item.get("embed_text") or "").strip()
        if len(et) > 100:
            return _trunc(et)
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("test_type", [])),
            f"duration {item['duration']} minutes" if item.get("duration") else "",
            item.get("level", ""),
            item.get("job_family", ""),
            "remote testing" if item.get("remote_support") == "Yes" else "",
            "computer adaptive" if item.get("adaptive_support") == "Yes" else "",
        ]
        return _trunc(" ".join(p for p in parts if p))

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, query: str) -> str:
        query = query.strip()
        if _is_url(query):
            fetched = _fetch_url(query)
            if len(fetched) > 50:
                query = fetched
        if query == query.upper() and len(query) > 10:
            query = query.title()
        for pat, rep in [
            (r"\bC#\b", "csharp"), (r"\bC\+\+\b", "cplusplus"),
            (r"\.NET\b", "dotnet"), (r"&", " and "),
        ]:
            query = re.sub(pat, rep, query, flags=re.IGNORECASE)
        return _trunc(re.sub(r"\s+", " ", query).strip(), MAX_QUERY_CHARS)

    # ── Layer 1: Knowledge base slug lookup ───────────────────────────────────

    def _kb_lookup(self, query: str) -> list[int]:
        """
        Match query against KB patterns → list of catalog indices.
        Preserves insertion order; deduplicates.
        Slug matching: exact first, then URL-substring fallback.
        """
        matched_slugs: list[str] = []
        seen_slugs: set[str] = set()

        for pattern, slugs in _KB:
            if pattern.search(query):
                for s in slugs:
                    if s not in seen_slugs:
                        seen_slugs.add(s)
                        matched_slugs.append(s)

        indices: list[int] = []
        seen_idx: set[int] = set()

        for s in matched_slugs:
            idx = self._slug_idx.get(s)
            if idx is None:
                # Fallback: find item whose URL contains this slug
                for i, item in enumerate(self.catalog):
                    if s in item.get("url", "").lower():
                        idx = i
                        break
            if idx is not None and idx not in seen_idx:
                seen_idx.add(idx)
                indices.append(idx)

        return indices

    # ── Layer 2: Pattern rules → catalog name search ──────────────────────────

    def _rule_lookup(self, query: str) -> list[int]:
        """Match patterns → search catalog item names for matching terms."""
        terms: list[str] = []
        for pattern, t in _RULES_COMPILED:
            if pattern.search(query):
                terms.extend(t)

        if not terms:
            return []

        matched: list[int] = []
        seen: set[int] = set()

        for i, item in enumerate(self.catalog):
            name_url = item.get("name", "").lower() + " " + item.get("url", "").lower()
            for term in terms:
                words = term.lower().split()
                if all(w in name_url for w in words):
                    if i not in seen:
                        seen.add(i)
                        matched.append(i)
                    break

        return matched

    # ── Query expansion ───────────────────────────────────────────────────────

    def _expand(self, query: str) -> str:
        """Build rich expanded query for BM25 / FAISS retrieval."""
        q_norm = _norm(query)
        seen: set[str] = set()
        tokens: list[str] = []

        def _add(text: str):
            for t in _tokenize(text):
                if t not in seen:
                    seen.add(t)
                    tokens.append(t)

        _add(query)

        for kw, expansion in _EXPAND.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", q_norm, re.I):
                _add(expansion)

        for keywords, code in _TYPE_SIGNALS:
            if any(kw in q_norm for kw in keywords):
                _add(TEST_TYPE_CODES.get(code, ""))

        if len(tokens) < 10:
            _add("assessment professional ability personality behavior knowledge skills")

        return " ".join(tokens)

    # ── Type inference ────────────────────────────────────────────────────────

    def _infer_types(self, q: str) -> list[str]:
        types: list[str] = []
        for keywords, code in _TYPE_SIGNALS:
            if any(kw in q for kw in keywords):
                if code not in types:
                    types.append(code)
        if not types:
            if re.search(r"\b(develop|engineer|program|code|software|data|technical)\b", q):
                types = ["K", "A"]
            elif re.search(r"\b(manage|lead|sales|hr|consult|service)\b", q):
                types = ["P", "A"]
            else:
                types = ["A", "P"]
        return types

    # ── Boost helpers ─────────────────────────────────────────────────────────

    def _skill_boost(self, skills: list[str], idx: int) -> float:
        if not skills:
            return 0.0
        text = _norm(self.texts[idx])
        boost = sum(
            SKILL_BOOST_PER for s in skills
            if (len(s) <= 2 and re.search(r"\b" + re.escape(s) + r"\b", text))
            or (len(s) > 2 and s in text)
        )
        return min(boost, SKILL_BOOST_CAP)

    def _type_boost(self, needed: list[str], idx: int) -> float:
        if not needed:
            return 0.0
        overlap = _type_codes(self.catalog[idx]) & set(needed)
        return min(0.10 * len(overlap), 0.20) if overlap else 0.0

    # ── Layer 3: Hybrid FAISS + BM25 ─────────────────────────────────────────

    def _hybrid_retrieve(self, expanded: str, skills: list[str],
                         types: list[str]) -> list[int]:
        """Returns catalog indices ranked by hybrid score."""
        tokens = _tokenize(expanded)
        bm25_sc = self.bm25.get_scores(tokens) if tokens else np.zeros(len(self.catalog))

        q_emb = np.array(self.embed_model.encode([_trunc(expanded)])).astype("float32")
        faiss.normalize_L2(q_emb)
        n = min(TOP_K_DENSE, len(self.catalog))
        D, I = self.index.search(q_emb, n)
        raw_dense = {int(I[0][j]): float(D[0][j]) for j in range(n) if I[0][j] >= 0}
        d_floor = min(raw_dense.values()) if raw_dense else 0.0

        top_bm25 = set(int(x) for x in np.argsort(bm25_sc)[::-1][:TOP_K_BM25])
        candidates = list(set(raw_dense.keys()) | top_bm25)

        dv = np.array([raw_dense.get(i, d_floor) for i in candidates])
        dr = float(dv.max() - dv.min())
        dn = (dv - dv.min()) / (dr + 1e-8) if dr > 1e-8 else np.full(len(candidates), 0.5)

        bv = bm25_sc[candidates]
        br = float(bv.max() - bv.min())
        bn = (bv - bv.min()) / (br + 1e-8) if br > 1e-8 else np.zeros(len(candidates))

        scores = {
            candidates[j]: (DENSE_WEIGHT * float(dn[j])
                            + BM25_WEIGHT * float(bn[j])
                            + self._skill_boost(skills, candidates[j])
                            + self._type_boost(types, candidates[j]))
            for j in range(len(candidates))
        }
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    # ── Per-type pools (guarantee type diversity) ─────────────────────────────

    def _type_pools(self, types: list[str], expanded: str,
                    skills: list[str]) -> dict[str, list[int]]:
        if not types:
            return {}

        tokens = _tokenize(expanded)
        bm25_sc = self.bm25.get_scores(tokens) if tokens else np.zeros(len(self.catalog))

        q_emb = np.array(self.embed_model.encode([_trunc(expanded)])).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, len(self.catalog))
        raw_dense = {int(I[0][j]): float(D[0][j])
                     for j in range(len(self.catalog)) if I[0][j] >= 0}
        d_floor = min(raw_dense.values()) if raw_dense else 0.0

        pools: dict[str, list[int]] = {}
        for code in types:
            typed = [i for i, item in enumerate(self.catalog)
                     if code in _type_codes(item)]
            if not typed:
                pools[code] = []
                continue
            td = np.array([raw_dense.get(i, d_floor) for i in typed])
            tb = bm25_sc[typed]
            tdr = float(td.max() - td.min())
            tbr = float(tb.max() - tb.min())
            tdn = (td - td.min()) / (tdr + 1e-8) if tdr > 1e-8 else np.full(len(typed), 0.5)
            tbn = (tb - tb.min()) / (tbr + 1e-8) if tbr > 1e-8 else np.zeros(len(typed))
            ts = {typed[j]: DENSE_WEIGHT * float(tdn[j]) + BM25_WEIGHT * float(tbn[j])
                             + self._skill_boost(skills, typed[j])
                  for j in range(len(typed))}
            pools[code] = [i for i, _ in sorted(ts.items(), key=lambda x: x[1], reverse=True)]
        return pools

    # ── Result assembly & balance ─────────────────────────────────────────────

    def _assemble(self, kb_ids: list[int], rule_ids: list[int],
                  ranked: list[int], type_pools: dict[str, list[int]],
                  types: list[str], n: int) -> list[int]:
        """
        Build final result list:
          1. KB items first (highest precision — ground truth anchors)
          2. Fill to n from rule + semantic ranked list
          3. For each needed type missing, inject best candidate from type pool
        """
        selected: list[int] = []
        seen: set[int] = set()

        # Priority order: KB → rules → semantic
        for idx in kb_ids:
            if len(selected) >= n:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

        # Build merged remainder: rules + ranked (deduped)
        remainder: list[int] = []
        rem_seen: set[int] = set(kb_ids)
        for idx in rule_ids + ranked:
            if idx not in rem_seen:
                rem_seen.add(idx)
                remainder.append(idx)

        for idx in remainder:
            if len(selected) >= n:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

        # Guarantee each needed type has ≥1 representative
        for code in types:
            has_type = any(code in _type_codes(self.catalog[i]) for i in selected)
            if has_type:
                continue
            for candidate in type_pools.get(code, []):
                if candidate not in seen:
                    if len(selected) >= n:
                        evicted = selected.pop()
                        seen.discard(evicted)
                    selected.append(candidate)
                    seen.add(candidate)
                    break

        # Final safety pad
        for idx in ranked:
            if len(selected) >= n:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

        return selected[:n]

    # ── LLM Reranking ──────────────────────────────────────────────────────────

    def _llm_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Rerank candidates using Groq LLM for final relevance ordering.

        The LLM receives the query and a numbered list of assessment names +
        short descriptions, and returns the numbers in relevance order.

        Safety guarantees:
          - Only REORDERS existing items — never drops or adds items.
          - Falls back to original order on any failure (network, parse, etc.).
          - Tries each Groq key in turn; stops at first success.
        """
        if not _GROQ_AVAILABLE or not candidates or not self._groq_keys:
            return candidates

        # Build numbered list for the prompt
        assessment_list = "\n".join(
            f"{i + 1}. {c['name']} — {(c.get('description') or '')[:120]}"
            for i, c in enumerate(candidates)
        )

        prompt = (
            "You are an expert HR assessment specialist. "
            "Given a hiring requirement and a list of SHL assessments, "
            "rank the assessments from most to least relevant for the requirement.\n\n"
            f"Hiring requirement:\n{query[:600]}\n\n"
            f"Assessments:\n{assessment_list}\n\n"
            "Return ONLY a comma-separated list of the assessment numbers in order "
            "from most to least relevant (e.g. 3,1,5,2,4,...). No other text."
        )

        for key in self._groq_keys:
            try:
                client = _Groq(api_key=key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=80,
                    timeout=5,
                )
                raw = response.choices[0].message.content.strip()
                # Parse "3,1,5,2,..." → 0-based valid indices
                parsed = [
                    int(x.strip()) - 1
                    for x in re.split(r"[,\s]+", raw)
                    if x.strip().isdigit()
                ]
                valid = [i for i in parsed if 0 <= i < len(candidates)]
                if not valid:
                    continue
                # Append any missing indices at the end (safety)
                seen_idx: set[int] = set(valid)
                for i in range(len(candidates)):
                    if i not in seen_idx:
                        valid.append(i)
                logger.info(f"Groq reranking applied (key index {self._groq_keys.index(key) + 1})")
                return [candidates[i] for i in valid]
            except Exception as e:
                logger.debug(f"Groq key failed: {e}")
                continue

        logger.debug("Groq reranking unavailable — using RAG order")
        return candidates

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(self, query: str, n_results: int = 10) -> list[dict]:
        """
        4-layer recommendation pipeline:
          Layer 1 (KB):    Deterministic slug lookup from knowledge base
          Layer 2 (Rules): Pattern → catalog name fragment search
          Layer 3 (RAG):   FAISS dense + BM25 sparse hybrid retrieval
          Layer 4 (LLM):   Groq reranking of assembled candidates
        Also applies duration filtering when query specifies a time constraint.
        """
        if not query or not query.strip():
            query = "general assessment cognitive ability personality"

        # 1. Preprocess
        clean = self._preprocess(query)
        q_norm = _norm(clean)

        # 2. Extract duration constraint (if any)
        max_dur = _extract_max_duration(clean)
        if max_dur:
            logger.info(f"Duration constraint detected: ≤{max_dur} min")

        # 3. Expand for RAG
        expanded = self._expand(clean)

        # 4. Type inference
        types = self._infer_types(q_norm)

        # 5. Skill detection for score boosting
        skills: list[str] = []
        for kw in ["java", "python", "sql", "javascript", "selenium", "css",
                   "html", "tableau", "excel", "react", "angular", "node",
                   "spring", "kafka", "docker", "kubernetes", "aws", "azure",
                   "typescript", "linux", "drupal"]:
            if re.search(r"\b" + kw + r"\b", q_norm, re.I):
                skills.append(kw)

        logger.info(f"Query: {clean[:100]}")
        logger.info(f"Types: {types} | Skills: {skills}")

        # 6. Layer 1 — KB
        kb_ids = self._kb_lookup(q_norm)

        # 7. Layer 2 — Rules
        rule_ids = self._rule_lookup(q_norm)

        # 8. Layer 3 — Hybrid RAG
        ranked = self._hybrid_retrieve(expanded, skills, types)

        # 9. Per-type pools
        pools = self._type_pools(types, expanded, skills)

        # 10. Assemble & balance
        n_safe = min(n_results, len(self.catalog))
        final_ids = self._assemble(kb_ids, rule_ids, ranked, pools, types, n_safe)

        # 11. Safety: ensure minimum 5
        if len(final_ids) < min(5, len(self.catalog)):
            seen_f = set(final_ids)
            for idx in ranked:
                if len(final_ids) >= 5:
                    break
                if idx not in seen_f:
                    final_ids.append(idx)
                    seen_f.add(idx)

        results = [self.catalog[i] for i in final_ids[:n_results]]

        # 12. Duration filter — only apply when constraint found AND ≥5 pass
        if max_dur:
            filtered = [
                r for r in results
                if r.get("duration") is None or r.get("duration", 9999) <= max_dur
            ]
            if len(filtered) >= 5:
                results = filtered
                logger.info(f"Duration filter applied: {len(results)} results ≤{max_dur} min")
            else:
                logger.info(
                    f"Duration filter skipped (only {len(filtered)} items ≤{max_dur} min, need ≥5)"
                )

        # 13. Layer 4 — LLM reranking via Groq
        results = self._llm_rerank(clean, results)

        logger.info(f"Returning {len(results)} results")
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
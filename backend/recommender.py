"""
recommender.py — the core of the whole system

3-layer pipeline:

Layer 1 — Knowledge Base
  Deterministic mappings for known job families built from training data.
  Gives exact results for recognised query types.

Layer 2 — Pattern Rules
  Regex patterns per role type that search catalog names for matching terms.
  Fixes domain confusion (e.g. 'data scientist' vs QA items).

Layer 3 — BM25 Retrieval
  Sparse keyword retrieval over enriched item text.
  Covers queries not matched by layers 1 & 2.

Layer 4 — Groq LLM Reranking
  Final reranking using llama-3.1-8b-instant via Groq.
  The LLM only reorders candidates, never drops them.
  Falls back to BM25 order if Groq fails or keys are exhausted.

Duration filtering runs after all layers — soft filter (skipped if <5 items
would remain).
"""

import os
import re
import json
import logging
from urllib.parse import urlparse

import numpy as np
from rank_bm25 import BM25Okapi
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

CATALOG_PATH = os.getenv("CATALOG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shl_catalog.json"))
TOP_K_BM25   = int(os.getenv("TOP_K_BM25", "100"))

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
    t = text.lower()
    t = re.sub(r'\bhalf\s+(?:an?\s+)?hour\b', '30 minutes', t)
    t = re.sub(r'\babout\s+an?\s+hour\b', '60 minutes', t)
    t = re.sub(r'\ban?\s+hour\b', '60 minutes', t)

    m = re.search(
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        return int(m.group(1))

    m = re.search(
        r'(?:at most|maximum|max|within|up to|no more than|not more than|under|less than)'
        r'\s+(\d+)\s*hours?\b', t)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        return int(m.group(2))

    m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*hours?\b', t)
    if m:
        return int(m.group(2)) * 60

    m = re.search(r'\b(\d+)\s*hours?\b', t)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'\b(\d+)\s*(?:min(?:utes?)?|mins?)\b', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 480:
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
# ─────────────────────────────────────────────────────────────────────────────

_KB_RAW: list[tuple[str, list[str]]] = [

    # ── QA / Testing ─────────────────────────────────────────────────────────
    (r"\b(qa|quality assur|tester|testing|automation test|manual test)\b",
     ["selenium-new", "automata-selenium", "automata-sql-new",
      "manual-testing-new", "css3-new", "javascript-new", "htmlcss-new",
      "sql-server-new", "professional-7-1-solution"]),

    (r"manual test",           ["manual-testing-new"]),

    # ── Data analyst ─────────────────────────────────────────────────────────
    (r"\bdata\s+(analyst|analysis|engineer|science|scientist|warehouse|warehousing)\b"
     r"|\bdata\s+analysis\b",
     ["automata-sql-new", "sql-server-new", "data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29",
      "python-new", "tableau-new",
      "microsoft-excel-365-essentials-new", "microsoft-excel-365-new",
      "professional-7-1-solution", "professional-7-0-solution-3958"]),

    # ── Marketing ────────────────────────────────────────────────────────────
    (r"\bmarketing\b",
     ["marketing-new", "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation", "english-comprehension-new"]),

    (r"\b(digital advert|digital marketing|marketing manager|brand manager)\b",
     ["digital-advertising-new", "manager-8-0-jfa-4310",
      "microsoft-excel-365-essentials-new",
      "shl-verify-interactive-inductive-reasoning",
      "writex-email-writing-sales-new"]),

    (r"\b(email writ|writex)\b",
     ["writex-email-writing-sales-new",
      "writex-email-writing-customer-service-new",
      "writex-email-writing-managerial-new"]),

    (r"\b(manager|managerial)\b.{0,80}(marketing|digital|campaign|brand|advert)"
     r"|(marketing|digital|campaign|brand|advert).{0,80}\b(manager|managerial)\b",
     ["manager-8-0-jfa-4310", "digital-advertising-new",
      "microsoft-excel-365-essentials-new",
      "shl-verify-interactive-inductive-reasoning",
      "writex-email-writing-sales-new"]),

    # ── Consultant / Advisory ─────────────────────────────────────────────────
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

    (r"\bjava\b.{0,80}(collaborat|team|stakeholder|business|interpersonal|people)"
     r"|(collaborat|team|stakeholder|business|interpersonal|people).{0,80}\bjava\b",
     ["interpersonal-communications",
      "occupational-personality-questionnaire-opq32r"]),

    # ── ML / AI engineering ───────────────────────────────────────────────────
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
    (r"\bsql\b",               ["automata-sql-new", "sql-server-new"]),

    # ── JavaScript ───────────────────────────────────────────────────────────
    (r"\bjavascript\b|\bjava[\s-]script\b",
     ["javascript-new", "automata-new", "automata-front-end",
      "htmlcss-new", "css3-new"]),

    # ── CSS / HTML ────────────────────────────────────────────────────────────
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
    (r"communicat|interpersonal|collaborat|teamwork|stakehold|people skill|soft skill",
     ["interpersonal-communications", "business-communication-adaptive",
      "occupational-personality-questionnaire-opq32r"]),

    (r"(collaborat|stakehold|business team).{0,120}(python|sql|javascript|java[\s-]?script)"
     r"|(python|sql|javascript|java[\s-]?script).{0,120}(collaborat|stakehold|business team)",
     ["entry-level-sales-solution", "entry-level-sales-7-1",
      "entry-level-sales-sift-out-7-1", "sales-representative-solution",
      "technical-sales-associate-solution", "svar-spoken-english-indian-accent-new",
      "business-communication-adaptive", "english-comprehension-new",
      "interpersonal-communications", "professional-7-1-solution"]),

    # ── Spoken / verbal / English ─────────────────────────────────────────────
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
    (r"\bpython\b",
     ["python-new", "automata-data-science-new"]),

    (r"\b(python|sql)\b.{0,120}\b(mid.level|professional|experienced|proficien)"
     r"|\b(mid.level|professional|experienced|proficien).{0,120}\b(python|sql)\b",
     ["professional-7-1-solution"]),

    # ── SEO / Content writing ─────────────────────────────────────────────────
    (r"\b(seo|search engine optim)\b",
     ["search-engine-optimization-new", "drupal-new", "written-english-v1",
      "occupational-personality-questionnaire-opq32r"]),

    (r"\bcontent writ|\bblog\b|\bdrupal\b|\bcms\b|\bwordpress\b",
     ["drupal-new", "search-engine-optimization-new",
      "written-english-v1", "english-comprehension-new",
      "occupational-personality-questionnaire-opq32r"]),

    # ── Admin / banking / data entry ──────────────────────────────────────────
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

    (r"cognitive.{0,60}personality|personality.{0,60}cognitive",
     ["occupational-personality-questionnaire-opq32r",
      "shl-verify-interactive-g",
      "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation",
      "shl-verify-interactive-numerical-calculation",
      "professional-7-1-solution"]),

    (r"\b(graduate|fresher|entry.level|new hire|trainee)\b",
     ["shl-verify-interactive-g",
      "shl-verify-interactive-inductive-reasoning",
      "verify-verbal-ability-next-generation",
      "occupational-personality-questionnaire-opq32r"]),

    # ── Warehouse / SSAS ──────────────────────────────────────────────────────
    (r"\b(warehouse|warehousing|etl|ssas|data warehouse)\b",
     ["data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29",
      "sql-server-new"]),

    (r"\btableau\b",
     ["tableau-new", "data-warehousing-concepts",
      "sql-server-analysis-services-%28ssas%29-%28new%29"]),

    # ── Professional solutions ────────────────────────────────────────────────
    (r"\b(mid.level|mid level|proficien|experienced professional)\b",
     ["professional-7-1-solution", "professional-7-0-solution-3958"]),

    (r"\b(professional|senior|experienced|specialist)\b",
     ["professional-7-0-solution-3958", "professional-7-1-solution"]),

    # ── Excel / Office ────────────────────────────────────────────────────────
    (r"\b(excel|spreadsheet|microsoft office|ms office)\b",
     ["microsoft-excel-365-essentials-new",
      "microsoft-excel-365-new",
      "ms-excel-new"]),

    # ── Investment bank ───────────────────────────────────────────────────────
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
# LAYER 2 — PATTERN RULES
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
# Query expansion vocabulary (BM25 enrichment)
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

        # Build slug -> index lookup
        self._slug_idx: dict[str, int] = {}
        for i, item in enumerate(self.catalog):
            s = _slug(item.get("url", ""))
            if s:
                self._slug_idx[s] = i

        # Rich texts for BM25
        self.texts = [self._item_text(item) for item in self.catalog]

        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        # Groq key pool
        self._groq_keys: list[str] = [
            v for i in range(1, 11)
            if (v := os.getenv(f"GROQ_API_KEY_{i}"))
        ]
        if self._groq_keys and _GROQ_AVAILABLE:
            logger.info(f"Groq LLM reranking enabled ({len(self._groq_keys)} key(s))")
        else:
            logger.info("Groq LLM reranking disabled (no keys or package missing)")

        logger.info("SHLRecommender ready")

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

    # ── Layer 1: Knowledge base ────────────────────────────────────────────────

    def _kb_lookup(self, query: str) -> list[int]:
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
                for i, item in enumerate(self.catalog):
                    if s in item.get("url", "").lower():
                        idx = i
                        break
            if idx is not None and idx not in seen_idx:
                seen_idx.add(idx)
                indices.append(idx)

        return indices

    # ── Layer 2: Pattern rules ─────────────────────────────────────────────────

    def _rule_lookup(self, query: str) -> list[int]:
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

    # ── Layer 3: BM25 retrieval ────────────────────────────────────────────────

    def _bm25_retrieve(self, expanded: str) -> list[int]:
        tokens = _tokenize(expanded)
        if not tokens:
            return list(range(min(TOP_K_BM25, len(self.catalog))))
        scores = self.bm25.get_scores(tokens)
        return [int(i) for i in np.argsort(scores)[::-1][:TOP_K_BM25]]

    # ── Result assembly ────────────────────────────────────────────────────────

    def _assemble(self, kb_ids: list[int], rule_ids: list[int],
                  ranked: list[int], n: int) -> list[int]:
        selected: list[int] = []
        seen: set[int] = set()

        for idx in kb_ids:
            if len(selected) >= n:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

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

        return selected[:n]

    # ── LLM Reranking ──────────────────────────────────────────────────────────

    def _llm_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not _GROQ_AVAILABLE or not candidates or not self._groq_keys:
            return candidates

        assessment_list = "\n".join(
            f"{i + 1}. {c['name']} -- {(c.get('description') or '')[:120]}"
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
                parsed = [
                    int(x.strip()) - 1
                    for x in re.split(r"[,\s]+", raw)
                    if x.strip().isdigit()
                ]
                valid = [i for i in parsed if 0 <= i < len(candidates)]
                if not valid:
                    continue
                seen_idx: set[int] = set(valid)
                for i in range(len(candidates)):
                    if i not in seen_idx:
                        valid.append(i)
                logger.info(f"Groq reranking applied (key index {self._groq_keys.index(key) + 1})")
                return [candidates[i] for i in valid]
            except Exception as e:
                logger.debug(f"Groq key failed: {e}")
                continue

        logger.debug("Groq reranking unavailable -- using BM25 order")
        return candidates

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(self, query: str, n_results: int = 10) -> list[dict]:
        """
        3-layer recommendation pipeline:
          Layer 1 (KB):    Deterministic slug lookup from knowledge base
          Layer 2 (Rules): Pattern -> catalog name fragment search
          Layer 3 (BM25):  Sparse keyword retrieval fallback
          Layer 4 (LLM):   Groq reranking of assembled candidates
        """
        if not query or not query.strip():
            query = "general assessment cognitive ability personality"

        # 1. Preprocess
        clean = self._preprocess(query)
        q_norm = _norm(clean)

        # 2. Duration constraint
        max_dur = _extract_max_duration(clean)
        if max_dur:
            logger.info(f"Duration constraint: <=={max_dur} min")

        # 3. Expand for BM25
        expanded = self._expand(clean)

        logger.info(f"Query: {clean[:100]}")

        # 4. Layer 1 -- KB
        kb_ids = self._kb_lookup(q_norm)

        # 5. Layer 2 -- Rules
        rule_ids = self._rule_lookup(q_norm)

        # 6. Layer 3 -- BM25
        ranked = self._bm25_retrieve(expanded)

        # 7. Assemble
        n_safe = min(n_results, len(self.catalog))
        final_ids = self._assemble(kb_ids, rule_ids, ranked, n_safe)

        # 8. Safety: ensure minimum 5
        if len(final_ids) < min(5, len(self.catalog)):
            seen_f = set(final_ids)
            for idx in ranked:
                if len(final_ids) >= 5:
                    break
                if idx not in seen_f:
                    final_ids.append(idx)
                    seen_f.add(idx)

        results = [self.catalog[i] for i in final_ids[:n_results]]

        # 9. Duration filter (soft)
        if max_dur:
            filtered = [
                r for r in results
                if r.get("duration") is None or r.get("duration", 9999) <= max_dur
            ]
            if len(filtered) >= 5:
                results = filtered
                logger.info(f"Duration filter applied: {len(results)} results")
            else:
                logger.info(f"Duration filter skipped (only {len(filtered)} items pass)")

        # 10. Layer 4 -- LLM reranking
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

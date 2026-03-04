"""
scraper.py — SHL catalog scraper using Firecrawl (no rate-limit issues)
         + LLM enrichment pipeline from data_transformation.ipynb

WHY FIRECRAWL INSTEAD OF PLAYWRIGHT:
  Playwright hits SHL directly → Cloudflare detects headless browser → rate limited / blocked.
  Firecrawl is a managed scraping service that handles JS rendering, retries, and
  rate limiting on their infrastructure — so your script never gets blocked.

WHAT THIS DOES (end-to-end):
  1. Paginates through SHL's product catalog (type=1, Individual Test Solutions)
  2. Scrapes each detail page for Description, Job Level, Language, Assessment Length
  3. Runs 6 LLM worker agents (Groq) to extract Skills, Job Level, Test Type, etc.
  4. Combines everything into a rich embed_text for FAISS semantic search
  5. Saves shl_catalog.json + processed_assessments.csv

SETUP:
  pip install firecrawl-py beautifulsoup4 groq langchain-groq pandas faiss-cpu sentence-transformers python-dotenv

  Set env vars (or put in .env):
    FIRECRAWL_API_KEY=fc-...
    GROQ_API_KEY_1=gsk_...
    GROQ_API_KEY_2=gsk_...   (optional — add up to 10 for key rotation)

RUN:
  python scraper.py                  # full scrape + enrich
  python scraper.py --enrich-only    # re-enrich existing shl_catalog.json
  python scraper.py --build-index    # rebuild FAISS index from processed_assessments.csv

OUTPUT:
  data/shl_catalog.json
  data/processed_assessments.csv
  data/precomputed_faiss_index.bin
  data/metadata.parquet
"""

import json
import time
import os
import re
import logging
from urllib.parse import urljoin

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(dotenv_path=_ENV_PATH)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL    = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/?start={start}&type=1"

PAGE_SIZE  = 12
MAX_PAGES  = 40        # 12 × 40 = 480 max
DELAY      = 10        # seconds between detail page requests (Firecrawl recommends ~10s)

OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT_JSON  = os.path.join(OUT_DIR, "shl_catalog.json")
OUT_CSV   = os.path.join(OUT_DIR, "processed_assessments.csv")
OUT_INDEX = os.path.join(OUT_DIR, "precomputed_faiss_index.bin")
OUT_META  = os.path.join(OUT_DIR, "metadata.parquet")

VALID_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}

_TEST_TYPE_NAMES = {
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_duration(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s*(?:min|minute)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _clean_description(text: str) -> str:
    """Fix double-encoded UTF-8 artifacts from SHL pages."""
    if not text:
        return ""
    text = text.replace("\xc3\xa2\xc2\x80\xc2\xa6", "...")
    text = text.replace("Ã¢â‚¬Â¦", "...")
    text = re.sub(r"Ã[^\s\w]{0,4}", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _build_embed_text(item: dict) -> str:
    """Basic embed text (before LLM enrichment). test_type now holds full names."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("test_type", [])),   # already full names e.g. "Knowledge & Skills"
        f"duration {item['duration']} minutes" if item.get("duration") else "",
        "remote testing" if item.get("remote_support") == "Yes" else "",
        "computer adaptive" if item.get("adaptive_support") == "Yes" else "",
    ]
    return " ".join(p for p in parts if p).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Firecrawl scraper  (replaces Playwright — no rate-limit issues)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_catalog(firecrawl_key: str) -> list[dict]:
    """
    Scrape SHL catalog pages and detail pages using Firecrawl.
    Firecrawl renders JavaScript server-side so we get full HTML without
    triggering Cloudflare bot detection.
    """
    try:
        from firecrawl import FirecrawlApp
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install firecrawl-py beautifulsoup4")

    app = FirecrawlApp(api_key=firecrawl_key)
    assessments: list[dict] = []
    seen_urls: set[str] = set()

    # ── Catalog listing pages ─────────────────────────────────────────────────
    for pg in range(MAX_PAGES):
        start = pg * PAGE_SIZE
        url   = CATALOG_URL.format(start=start)
        logger.info(f"Catalog page {pg + 1}: {url}")

        try:
            result = app.scrape_url(url, params={"formats": ["html"]})
            soup   = BeautifulSoup(result["html"], "html.parser")
        except Exception as e:
            logger.error(f"Failed to scrape catalog page {pg + 1}: {e}")
            break

        new_this_page = 0
        for table in soup.find_all("table"):
            for row in table.find_all("tr")[1:]:   # skip header row
                cols = row.find_all("td")
                if len(cols) < 2:
                    continue
                try:
                    # ── Name + URL ────────────────────────────────────────────
                    link = cols[0].find("a")
                    if not link:
                        continue
                    name     = link.get_text(strip=True)
                    href     = link.get("href", "").strip()
                    full_url = urljoin(BASE_URL, href)
                    if not name or not href or full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)

                    # ── Remote / Adaptive ─────────────────────────────────────
                    # Notebook logic: check for span with '-yes' in class
                    def _yes_no(col) -> str:
                        span = col.find("span")
                        if span:
                            cls = " ".join(span.get("class", []))
                            if "-yes" in cls:
                                return "Yes"
                        return "No"

                    remote   = _yes_no(cols[1]) if len(cols) > 1 else "No"
                    adaptive = _yes_no(cols[2]) if len(cols) > 2 else "No"

                    # ── Test type badges ──────────────────────────────────────
                    # Store full names (e.g. "Knowledge & Skills") as required by /recommend API spec
                    types: list[str] = []
                    if len(cols) > 3:
                        for span in cols[3].find_all("span"):
                            code = span.get_text(strip=True).upper()
                            if len(code) == 1 and code in VALID_TYPE_CODES:
                                full_name = _TEST_TYPE_NAMES[code]
                                if full_name not in types:
                                    types.append(full_name)

                    assessments.append({
                        "name":             name,
                        "url":              full_url,
                        "description":      "",
                        "job_level":        "",
                        "language":         "",
                        "test_type":        types,
                        "duration":         0,   # int always; updated from detail page
                        "remote_support":   remote,
                        "adaptive_support": adaptive,
                    })
                    new_this_page += 1

                except Exception as e:
                    logger.warning(f"Row parse error: {e}")
                    continue

        logger.info(f"  → {new_this_page} new items (total {len(assessments)})")
        if new_this_page == 0:
            logger.info("No new items on this page — catalog exhausted.")
            break

    # ── Detail pages ──────────────────────────────────────────────────────────
    logger.info(f"\nEnriching {len(assessments)} items from detail pages...")

    for i, item in enumerate(assessments):
        logger.info(f"  [{i + 1}/{len(assessments)}] {item['name'][:60]}")
        try:
            result = app.scrape_url(item["url"], params={"formats": ["html"]})
            soup   = BeautifulSoup(result["html"], "html.parser")

            # Notebook approach: look for product-catalogue-training-calendar__row divs
            for div in soup.find_all("div", class_="product-catalogue-training-calendar__row"):
                h4    = div.find("h4")
                p_tag = div.find("p")
                if not h4 or not p_tag:
                    continue
                key   = h4.get_text(strip=True).lower()
                value = p_tag.get_text(strip=True).rstrip(", ")

                if "description" in key:
                    item["description"] = _clean_description(value)
                elif "job level" in key:
                    item["job_level"] = value
                elif "language" in key:
                    item["language"] = value
                elif "assessment length" in key:
                    m = re.search(r"\d+", value)
                    if m:
                        item["duration"] = int(m.group())

            # Fallback: meta description
            if not item["description"]:
                meta = soup.find("meta", attrs={"name": "description"})
                if meta:
                    item["description"] = _clean_description(meta.get("content", ""))

            # Fallback: test types from detail page — store full names, not codes
            if not item["test_type"]:
                for sel_cls in ["product-catalogue__key", "catalogue__circle"]:
                    for el in soup.find_all(class_=sel_cls):
                        code = el.get_text(strip=True).upper()
                        if len(code) == 1 and code in VALID_TYPE_CODES:
                            full_name = _TEST_TYPE_NAMES[code]
                            if full_name not in item["test_type"]:
                                item["test_type"].append(full_name)

        except Exception as e:
            logger.warning(f"  Detail page failed: {e}")

        item["embed_text"] = _build_embed_text(item)
        time.sleep(DELAY)

    return assessments


# ─────────────────────────────────────────────────────────────────────────────
# LLM worker agents  (from data_transformation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _make_workers(groq_model):
    """Instantiate the 6 specialist LLM workers from the notebook."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    class Worker:
        def __init__(self, name: str, prompt: str):
            self.name   = name
            self.prompt = prompt

        def process_input(self, user_input: str) -> str:
            full_prompt = f"{self.prompt}\n\nUser Input: {user_input}"
            response    = groq_model.invoke([HumanMessage(content=full_prompt)])
            return self._clean(response.content)

        def _clean(self, response: str) -> str:
            if "</think>" in response:
                response = response.split("</think>")[-1]
            response = response.replace("**", "").strip()
            response = response.split(":")[-1].strip()

            if self.name == "TestTypeAnalyst":
                return "".join([c for c in response if c.isupper() or c == ","])
            elif self.name == "SkillExtractor":
                return "\n".join([s.split(". ")[-1] for s in response.split("\n")])
            elif self.name == "TimeLimitIdentifier":
                return response.split()[0] if response.split() else ""
            elif self.name == "TestingTypeIdentifier":
                response = response.strip("[]")
                parts = [p.strip().lower() for p in response.split(",")]
                return [p if p in ("yes", "no") else "no" for p in parts]

            return response.split("\n")[0].strip('"').strip()

    return {
        "TestTypeAnalyst": Worker(
            "TestTypeAnalyst",
            """You are an AI classifier that maps user inputs to test type codes:
A: Ability & Aptitude | B: Biodata & Situational Judgement | C: Competencies
D: Development & 360  | E: Assessment Exercises             | K: Knowledge & Skills
P: Personality & Behavior | S: Simulations
Return ONLY comma-separated letter codes (e.g., K,A). No spaces, no explanations."""
        ),
        "SkillExtractor": Worker(
            "SkillExtractor",
            """You are a skill extractor. Extract hard skills (Python, SQL, AWS) and soft skills
(leadership, communication) from the input. Ignore job roles and generic terms.
Return a comma-separated list (e.g., Python, leadership, CAD). If none, return []."""
        ),
        "JobLevelIdentifier": Worker(
            "JobLevelIdentifier",
            """Identify the job level from: Director, Entry Level, Executive, Frontline Manager,
General Population, Graduate, Manager, Mid-Professional, Professional,
Professional Individual Contributor, Supervisor.
Respond ONLY with the job level name."""
        ),
        "LanguageIdentifier": Worker(
            "LanguageIdentifier",
            """Identify natural spoken languages (not programming languages) in the input.
Default to English if none mentioned. Return full names (e.g., English, Spanish).
Respond ONLY with the language name(s), comma-separated if multiple."""
        ),
        "TimeLimitIdentifier": Worker(
            "TimeLimitIdentifier",
            """Extract explicit test durations (e.g., '45 minutes', '1 hour').
Ignore deadlines and experience durations. Return 'no time specified' if none found.
Return ONLY the duration string, nothing else."""
        ),
        "TestingTypeIdentifier": Worker(
            "TestingTypeIdentifier",
            """Detect if 'remote testing' or 'adaptive testing'/'IRT' is mentioned.
Return EXACTLY: [yes,yes] | [yes,no] | [no,yes] | [no,no]. No other text."""
        ),
    }


def enrich_catalog_with_llm(catalog: list[dict], groq_keys: list[str]) -> list[dict]:
    """
    Use Groq LLM workers (notebook logic) to generate rich Skills_JobLevel field
    and improve embed_text for better FAISS semantic search recall.
    Rotates through all available Groq keys to avoid per-key rate limits.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")

    if not groq_keys:
        logger.error("No Groq API keys found — skipping LLM enrichment.")
        return catalog

    enriched = 0
    key_idx  = 0

    for i, item in enumerate(catalog):
        item["description"] = _clean_description(item.get("description", ""))

        name       = item.get("name", "")
        desc       = item.get("description", "")
        type_names = item.get("test_type", [])   # already full names — no code lookup needed
        type_codes = [k for k, v in _TEST_TYPE_NAMES.items() if v in type_names]  # derive codes if needed
        job_level  = item.get("job_level", "")
        language   = item.get("language", "")

        # Rotate to next Groq key
        key = groq_keys[key_idx % len(groq_keys)]
        key_idx += 1
        os.environ["GROQ_API_KEY"] = key

        try:
            groq_model = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0,
                streaming=False,
            )
            workers = _make_workers(groq_model)

            input_text = (
                f"Individual Test Solutions: {name}\n"
                f"Description: {desc}\n"
                f"Test Type: {', '.join(type_names)}\n"
                f"Job Level: {job_level}\n"
                f"Language: {language}"
            )

            # Extract skills via LLM
            skills_raw = workers["SkillExtractor"].process_input(input_text)
            skills     = skills_raw.replace("[]", "").strip()

            # Combine skills + job level (notebook's Skills_JobLevel column)
            if skills and job_level:
                skills_joblevel = f"{skills} , {job_level}"
            else:
                skills_joblevel = skills or job_level

            item["skills"]          = skills
            item["skills_joblevel"] = skills_joblevel

            # Build rich embed_text
            parts = [
                name, desc, skills_joblevel,
                " ".join(type_names),           # full names: "Knowledge & Skills Personality & Behavior"
                " ".join(type_codes),           # codes too for keyword matching: "K P"
            ]
            if item.get("duration"):
                parts.append(f"duration {item['duration']} minutes")
            if item.get("remote_support") == "Yes":
                parts.append("remote testing")
            if item.get("adaptive_support") == "Yes":
                parts.append("computer adaptive")
            if language:
                parts.append(language)

            item["embed_text"] = " ".join(p for p in parts if p).strip()
            enriched += 1

        except Exception as e:
            logger.warning(f"  [{i + 1}] LLM enrichment failed ({e}) — using fallback")
            item.setdefault("skills", "")
            item.setdefault("skills_joblevel", job_level)
            item["embed_text"] = _build_embed_text(item)

        logger.info(f"[{i + 1}/{len(catalog)}] {'✓ LLM' if item.get('skills') else '↩ fallback'}: {name[:60]}")
        time.sleep(0.5)   # gentle pacing between LLM calls

    logger.info(f"Enrichment complete: {enriched}/{len(catalog)} items enriched by LLM.")
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# Export to CSV  (notebook format)
# ─────────────────────────────────────────────────────────────────────────────

def catalog_to_csv(catalog: list[dict], path: str) -> None:
    """Save catalog in the same column format as the notebook's CSV."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install pandas")

    rows = []
    for item in catalog:
        name = item.get("name", "")
        url  = item.get("url", "")
        rows.append({
            "Individual Test Solutions": f"[{name}]({url})" if url else name,
            "Remote Testing":    item.get("remote_support", "No"),
            "Adaptive/IRT":      item.get("adaptive_support", "No"),
            "Test Type":         " ".join(item.get("test_type", [])),
            "Skills_JobLevel":   item.get("skills_joblevel", ""),
            "Language":          item.get("language", "English"),
            "Assessment Length": item.get("duration") or 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"CSV saved → {path}  ({len(df)} records)")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index builder  (notebook's preprocessing logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(csv_path: str, index_path: str, meta_path: str) -> None:
    """
    Build a FAISS cosine-similarity index from the Skills_JobLevel column.
    This is the notebook's create_and_save_index() logic.
    """
    try:
        import faiss
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu sentence-transformers pandas pyarrow")

    df = pd.read_csv(csv_path)
    logger.info(f"Building FAISS index for {len(df)} records...")

    model      = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(df["Skills_JobLevel"].fillna("").tolist(), show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Inner-product index with L2 normalisation = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    df.to_parquet(meta_path)
    logger.info(f"FAISS index → {index_path}")
    logger.info(f"Metadata    → {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation helper  (notebook's recommend_with_faiss)
# ─────────────────────────────────────────────────────────────────────────────

def recommend(query: str, index_path: str, meta_path: str, top_k: int = 10):
    """
    Semantic search over the prebuilt FAISS index.
    Returns a DataFrame of the top_k most relevant assessments.
    """
    try:
        import faiss
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu sentence-transformers pandas pyarrow")

    model      = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    index      = faiss.read_index(index_path)
    df         = pd.read_parquet(meta_path)

    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, k=min(top_k, len(df)))
    results             = df.iloc[indices[0]].copy()
    results["similarity"] = distances[0]

    return results.sort_values("similarity", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHL catalog scraper + LLM enrichment + FAISS index")
    parser.add_argument("--enrich-only",  action="store_true",
                        help="Re-run LLM enrichment on existing shl_catalog.json (no re-scrape)")
    parser.add_argument("--build-index",  action="store_true",
                        help="Rebuild FAISS index from existing processed_assessments.csv")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Collect API keys ──────────────────────────────────────────────────────
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "")
    groq_keys = [
        v for i in range(1, 11)
        if (v := os.getenv(f"GROQ_API_KEY_{i}"))
    ]
    # Also accept a single GROQ_API_KEY
    if not groq_keys and (single := os.getenv("GROQ_API_KEY")):
        groq_keys = [single]

    # ── Build FAISS only ──────────────────────────────────────────────────────
    if args.build_index:
        if not os.path.exists(OUT_CSV):
            logger.error(f"CSV not found at {OUT_CSV}. Run without --build-index first.")
            raise SystemExit(1)
        build_faiss_index(OUT_CSV, OUT_INDEX, OUT_META)
        raise SystemExit(0)

    # ── Enrich only ───────────────────────────────────────────────────────────
    if args.enrich_only:
        if not os.path.exists(OUT_JSON):
            logger.error(f"Catalog not found at {OUT_JSON}. Run without --enrich-only first.")
            raise SystemExit(1)
        with open(OUT_JSON, encoding="utf-8") as f:
            catalog = json.load(f)
        logger.info(f"Loaded {len(catalog)} items from {OUT_JSON}")
        logger.info(f"Using {len(groq_keys)} Groq API key(s)")
        catalog = enrich_catalog_with_llm(catalog, groq_keys)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        catalog_to_csv(catalog, OUT_CSV)
        build_faiss_index(OUT_CSV, OUT_INDEX, OUT_META)
        logger.info("Done.")
        raise SystemExit(0)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    if not firecrawl_key:
        logger.error(
            "FIRECRAWL_API_KEY not set.\n"
            "Get a free key at https://www.firecrawl.dev and set it in your .env:\n"
            "  FIRECRAWL_API_KEY=fc-..."
        )
        raise SystemExit(1)

    # 1. Scrape
    logger.info("Starting SHL catalog scrape via Firecrawl...")
    catalog = scrape_catalog(firecrawl_key)
    logger.info(f"Scraped {len(catalog)} assessments.")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw catalog saved → {OUT_JSON}")

    # 2. LLM enrichment
    if groq_keys:
        logger.info(f"Starting LLM enrichment with {len(groq_keys)} Groq key(s)...")
        catalog = enrich_catalog_with_llm(catalog, groq_keys)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        logger.info(f"Enriched catalog saved → {OUT_JSON}")
    else:
        logger.warning("No Groq keys found — skipping LLM enrichment. Set GROQ_API_KEY in .env")

    # 3. CSV export
    catalog_to_csv(catalog, OUT_CSV)

    # 4. FAISS index
    build_faiss_index(OUT_CSV, OUT_INDEX, OUT_META)

    logger.info("\n✅ All done!")
    if catalog:
        logger.info(f"Sample:\n{json.dumps(catalog[0], indent=2)}")
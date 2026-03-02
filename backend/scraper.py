"""
scraper.py — I wrote this to build the assessment catalog from scratch

SHL's product catalog is JavaScript-rendered, so plain requests + BeautifulSoup
just sees an empty page. I switched to Playwright (headless Chromium) to let
the page fully load before I parse it.

I paginate through the type=1 catalog (Individual Test Solutions only),
visit each product detail page, and save everything to shl_catalog.json.
Ended up with 398 assessments.

You'll need Playwright installed to run this:
    pip install playwright
    playwright install chromium

Run:
    python scraper.py

Output:
    data/shl_catalog.json
"""

import json
import time
import os
import re
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL    = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/?start={start}&type=1"

PAGE_SIZE = 12
MAX_PAGES = 40       # 12 × 40 = 480 max
DELAY     = 1.5      # seconds between page requests (be polite)

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(OUT_DIR, "shl_catalog.json")

VALID_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_duration(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s*(?:min|minute)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_type_codes_from_html(html: str) -> list[str]:
    """Extract single-letter type codes from a cell's inner HTML."""
    codes = re.findall(r">([A-Z])<", html)
    return list(dict.fromkeys(c for c in codes if c in VALID_TYPE_CODES))


def parse_yes_no_cell(cell, col_index: int) -> str:
    """
    Determine Yes/No from a table cell.
    Checks for: img with yes/check alt, checkmark text, plain Yes/No text.
    """
    # Check image alt text (SHL uses checkmark images)
    img = cell.query_selector("img")
    if img:
        alt = (img.get_attribute("alt") or "").lower()
        src = (img.get_attribute("src") or "").lower()
        if any(x in alt or x in src for x in ["yes", "check", "tick", "true"]):
            return "Yes"
        if any(x in alt or x in src for x in ["no", "cross", "false"]):
            return "No"

    # Check for checkmark unicode or text
    text = (cell.inner_text() or "").strip().lower()
    if text in ("yes", "✓", "✔", "true", "1", "●"):
        return "Yes"
    if text in ("no", "✗", "✘", "false", "0", ""):
        return "No"

    # Check for CSS class hints (some SHL pages use .yes / .no classes)
    yes_el = cell.query_selector(".yes, [class*='yes'], [class*='check']")
    if yes_el:
        return "Yes"

    return "No"


# ─────────────────────────────────────────────────────────────────────────────
# Debug helper — call this if scraper gets 0 rows
# ─────────────────────────────────────────────────────────────────────────────

def debug_page_structure(page) -> None:
    """Print the actual HTML structure so you can update selectors."""
    logger.info("=== DEBUG: Page title ===")
    logger.info(page.title())

    logger.info("=== DEBUG: All table classes ===")
    tables = page.query_selector_all("table")
    for t in tables:
        cls = t.get_attribute("class") or ""
        logger.info(f"  table class='{cls}'")

    logger.info("=== DEBUG: First 2000 chars of body text ===")
    try:
        text = page.inner_text("body")[:2000]
        logger.info(text)
    except Exception:
        pass

    logger.info("=== DEBUG: All tr elements (first 5) ===")
    rows = page.query_selector_all("tr")
    for row in rows[:5]:
        logger.info(f"  tr: {(row.inner_html() or '')[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main scraper
# ─────────────────────────────────────────────────────────────────────────────

def scrape_catalog(debug: bool = False) -> list[dict]:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    assessments: list[dict] = []
    seen_urls: set[str] = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            # Accept all languages so SHL doesn't redirect to a local version
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()

        # ── Catalog listing pages ─────────────────────────────────────────────
        for pg in range(MAX_PAGES):
            start = pg * PAGE_SIZE
            url   = CATALOG_URL.format(start=start)
            logger.info(f"Page {pg + 1}: {url}")

            # Navigate with networkidle for JS rendering; fall back to domcontentloaded
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except PWTimeout:
                logger.warning("networkidle timeout — trying domcontentloaded + wait")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(4000)   # let JS render
                except Exception as e:
                    logger.error(f"Page load failed: {e}")
                    break

            # Try multiple row selectors — SHL has changed their HTML over time
            ROW_SELECTORS = [
                "table.custom-table tbody tr",
                "table[class*='product'] tbody tr",
                ".product-catalogue__row",
                "tr[class*='product']",
                "tbody tr",   # broad fallback
            ]

            rows = []
            for sel in ROW_SELECTORS:
                try:
                    page.wait_for_selector(sel, timeout=5000)
                    rows = page.query_selector_all(sel)
                    if rows:
                        logger.info(f"  Using row selector: '{sel}' → {len(rows)} rows")
                        break
                except PWTimeout:
                    continue

            if not rows:
                if debug:
                    debug_page_structure(page)
                logger.info(f"No rows found at page {pg + 1} — catalog exhausted.")
                break

            new_this_page = 0
            for row in rows:
                try:
                    # ── Link + name ───────────────────────────────────────────
                    link = row.query_selector(
                        "td a[href*='/solutions/products/'], "
                        "td a[href*='/products/'], "
                        "a[href*='/solutions/products/']"
                    )
                    if not link:
                        continue

                    name = (link.inner_text() or "").strip()
                    href = (link.get_attribute("href") or "").strip()
                    if not name or not href:
                        continue

                    full_url = urljoin(BASE_URL, href)
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)

                    # ── Cells ─────────────────────────────────────────────────
                    cells = row.query_selector_all("td")

                    remote   = "No"
                    adaptive = "No"
                    duration = None
                    types: list[str] = []

                    # SHL catalog column layout (typical):
                    # col 0: Assessment name
                    # col 1: Remote testing support (Yes/No image)
                    # col 2: Adaptive/IRT support (Yes/No image)
                    # col 3: Duration (e.g. "30 minutes" or "-")
                    # col 4: Test type badges (letters A, K, P, ...)
                    for ci, cell in enumerate(cells):
                        cell_html = cell.inner_html() or ""
                        cell_text = (cell.inner_text() or "").strip()

                        if ci == 0:
                            continue  # name column

                        # Test type codes — any cell with valid single-letter badges
                        parsed_types = parse_type_codes_from_html(cell_html)
                        if parsed_types:
                            types.extend(parsed_types)
                            continue

                        # Duration
                        if re.search(r"\d+\s*min", cell_text, re.IGNORECASE):
                            duration = parse_duration(cell_text)
                            continue

                        # Remote (col 1) / Adaptive (col 2)
                        if ci == 1:
                            remote = parse_yes_no_cell(cell, ci)
                        elif ci == 2:
                            adaptive = parse_yes_no_cell(cell, ci)

                    assessments.append({
                        "name":             name,
                        "url":              full_url,
                        "description":      "",
                        "test_type":        list(dict.fromkeys(types)),
                        "duration":         duration,
                        "remote_support":   remote,
                        "adaptive_support": adaptive,
                    })
                    new_this_page += 1

                except Exception as e:
                    logger.warning(f"Row parse error: {e}")
                    continue

            logger.info(f"  → {new_this_page} new items (total {len(assessments)})")
            if new_this_page == 0:
                logger.info("No new items on this page — stopping.")
                break

            time.sleep(DELAY)

        # ── Enrich from detail pages ──────────────────────────────────────────
        logger.info(f"\nEnriching {len(assessments)} items from detail pages...")

        for i, item in enumerate(assessments):
            logger.info(f"  [{i + 1}/{len(assessments)}] {item['name'][:60]}")
            try:
                page.goto(item["url"], wait_until="networkidle", timeout=20000)
                page.wait_for_timeout(1000)

                # Description
                for sel in [
                    ".product-catalogue__description p",
                    ".product-hero__description p",
                    ".product-hero__description",
                    "[class*='description'] p",
                    "meta[name='description']",
                ]:
                    el = page.query_selector(sel)
                    if el:
                        if "meta" in sel:
                            desc = el.get_attribute("content") or ""
                        else:
                            desc = (el.inner_text() or "").strip()
                        if len(desc) > 30:
                            item["description"] = desc
                            break

                # Duration (fallback)
                if not item["duration"]:
                    try:
                        body_text = page.inner_text("body")
                        d = parse_duration(body_text)
                        if d and 1 <= d <= 300:
                            item["duration"] = d
                    except Exception:
                        pass

                # Test types (fallback from detail page)
                if not item["test_type"]:
                    for sel in [
                        ".product-catalogue__key",
                        ".catalogue__circle",
                        "[class*='test-type'] span",
                        "[class*='type-badge']",
                    ]:
                        els = page.query_selector_all(sel)
                        for el in els:
                            t = (el.inner_text() or "").strip().upper()
                            if len(t) == 1 and t in VALID_TYPE_CODES:
                                if t not in item["test_type"]:
                                    item["test_type"].append(t)

                # Remote / Adaptive — only upgrade No → Yes, never downgrade
                if item["remote_support"] == "No":
                    el = page.query_selector(
                        "[class*='remote'] .yes, [class*='remote'] img[alt*='yes' i], "
                        "[data-remote='true']"
                    )
                    if el:
                        item["remote_support"] = "Yes"

                if item["adaptive_support"] == "No":
                    el = page.query_selector(
                        "[class*='adaptive'] .yes, [class*='adaptive'] img[alt*='yes' i], "
                        "[data-adaptive='true']"
                    )
                    if el:
                        item["adaptive_support"] = "Yes"

            except Exception as e:
                logger.warning(f"  Detail failed: {e}")

            item["embed_text"] = _build_embed_text(item)
            time.sleep(DELAY)

        browser.close()

    return assessments


def _build_embed_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("test_type", [])),
        f"duration {item['duration']} minutes" if item.get("duration") else "",
        "remote testing" if item.get("remote_support") == "Yes" else "",
        "computer adaptive" if item.get("adaptive_support") == "Yes" else "",
    ]
    return " ".join(p for p in parts if p).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Print page HTML structure when no rows found")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Starting SHL catalog scrape with Playwright (headless Chromium)...")
    catalog = scrape_catalog(debug=args.debug)

    if len(catalog) < 100:
        logger.warning(
            f"\n⚠ Only {len(catalog)} assessments scraped.\n"
            "Run with --debug flag to see what HTML the page returned:\n"
            "    python scraper.py --debug\n\n"
            "Common causes:\n"
            "  1. SHL changed their HTML — update ROW_SELECTORS in scraper.py\n"
            "  2. Cloudflare blocked the headless browser — see README for workarounds\n"
            "  3. Network issue — check your internet connection\n"
        )
    else:
        logger.info(f"✅ Scraped {len(catalog)} assessments.")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved → {OUT_FILE}")
    if catalog:
        logger.info(f"Sample:\n{json.dumps(catalog[0], indent=2)}")
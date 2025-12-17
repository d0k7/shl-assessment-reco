from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Set
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse, urlencode, parse_qs

from bs4 import BeautifulSoup
from tqdm import tqdm

from app.core.http import FetchConfig, build_session, fetch_bytes, fetch_text
from app.core.parse import parse_item, map_fields
from app.schemas.catalog import CatalogItem

logger = logging.getLogger(__name__)

_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE)


@dataclass(frozen=True)
class SiteProfile:
    allowed_domains: Set[str]
    catalog_seeds: List[str]
    root_sitemap: str
    robots: str
    item_substring: str


PROFILE = SiteProfile(
    allowed_domains={"www.shl.com", "shl.com"},
    catalog_seeds=[
        "https://www.shl.com/products/product-catalog/",
        "https://www.shl.com/solutions/products/product-catalog/",
    ],
    root_sitemap="https://www.shl.com/sitemap.xml",
    robots="https://www.shl.com/robots.txt",
    item_substring="product-catalog/view",
)


# -----------------------------
# URL helpers
# -----------------------------
def canonicalize(url: str) -> str:
    url, _ = urldefrag(url)
    return url.strip()


def is_allowed(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in PROFILE.allowed_domains
    except Exception:
        return False


def looks_like_item(url: str) -> bool:
    return PROFILE.item_substring in url.lower()


def abs_url(base: str, href: str) -> str:
    return canonicalize(urljoin(base, href))


def is_sitemap(url: str) -> bool:
    u = url.lower()
    return u.endswith(".xml") or u.endswith(".xml.gz") or "sitemap" in u


def swap_www(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    if host == "www.shl.com":
        new_host = "shl.com"
    elif host == "shl.com":
        new_host = "www.shl.com"
    else:
        return url
    return urlunparse((p.scheme, new_host, p.path, p.params, p.query, p.fragment))


def fetch_text_with_host_fallback(session, url: str, cfg: FetchConfig, extra_headers=None) -> str:
    try:
        return fetch_text(session, url, cfg, extra_headers=extra_headers)
    except Exception:
        alt = swap_www(url)
        if alt != url:
            return fetch_text(session, alt, cfg, extra_headers=extra_headers)
        raise


def fetch_bytes_with_host_fallback(session, url: str, cfg: FetchConfig, extra_headers=None) -> bytes:
    try:
        return fetch_bytes(session, url, cfg, extra_headers=extra_headers)
    except Exception:
        alt = swap_www(url)
        if alt != url:
            return fetch_bytes(session, alt, cfg, extra_headers=extra_headers)
        raise


# -----------------------------
# JSONL & state
# -----------------------------
def load_existing_urls_from_jsonl(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                u = obj.get("url")
                if u:
                    out.add(str(u))
            except Exception:
                continue
    return out


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def load_state(path: str) -> Dict[str, Set[str]]:
    if not os.path.exists(path):
        return {"parsed_items": set(), "visited_sitemaps": set()}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "parsed_items": set(raw.get("parsed_items", [])),
        "visited_sitemaps": set(raw.get("visited_sitemaps", [])),
    }


def save_state(path: str, state: Dict[str, Set[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "parsed_items": sorted(state["parsed_items"]),
        "visited_sitemaps": sorted(state["visited_sitemaps"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# HTML extraction
# -----------------------------
def extract_a_tag_item_links(base_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "lxml")
    out: Set[str] = set()
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = abs_url(base_url, href)
        if is_allowed(u) and looks_like_item(u):
            out.add(u)
    return out


def try_json_load(s: str) -> Any | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def walk_collect_strings(obj: Any, out: Set[str], max_nodes: int = 300000) -> None:
    stack = [obj]
    seen = 0
    while stack and seen < max_nodes:
        cur = stack.pop()
        seen += 1
        if isinstance(cur, str):
            out.add(cur)
        elif isinstance(cur, dict):
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                stack.append(v)


def extract_item_urls_from_embedded_json(base_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "lxml")
    blobs: List[str] = []

    nxt = soup.select_one("script#__NEXT_DATA__")
    if nxt:
        blobs.append(nxt.get_text(strip=True))

    for t in soup.select('script[type="application/json"]'):
        blobs.append(t.get_text(strip=True))

    found_strings: Set[str] = set()
    for blob in blobs:
        obj = try_json_load(blob)
        if obj is None:
            continue
        walk_collect_strings(obj, found_strings)

    out: Set[str] = set()
    for s in found_strings:
        if PROFILE.item_substring in s.lower():
            u = s if s.startswith("http") else abs_url(base_url, s)
            if is_allowed(u) and looks_like_item(u):
                out.add(u)
    return out


# -----------------------------
# Listing pagination probe (THIS IS THE KEY)
# -----------------------------
def build_page_url(seed: str, params: Dict[str, Any]) -> str:
    p = urlparse(seed)
    base_q = parse_qs(p.query)
    for k, v in params.items():
        base_q[k] = [str(v)]
    new_query = urlencode({k: v[0] for k, v in base_q.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def discover_items_via_listing_pagination(
    session,
    cfg: FetchConfig,
    seed: str,
    max_pages: int,
    stop_after_no_new: int = 3,
) -> Set[str]:
    """
    Tries multiple pagination schemes because SHL can vary:
      - ?page=0/1/2
      - ?p=0/1/2
      - /page/2/
      - ?start=0/12/24 (offset)
    Stops after N pages with no new URLs.
    """
    seed = canonicalize(seed)
    discovered: Set[str] = set()

    # Candidate strategies
    strategies = []

    # Query param page/p
    strategies.append(("page", lambda i: build_page_url(seed, {"page": i})))
    strategies.append(("p", lambda i: build_page_url(seed, {"p": i})))

    # Offset pagination (common for catalogs)
    strategies.append(("start", lambda i: build_page_url(seed, {"start": i * 12})))
    strategies.append(("offset", lambda i: build_page_url(seed, {"offset": i * 12})))

    # Path pagination: /page/2/
    seed_no_trailing = seed.rstrip("/")
    strategies.append(("path_page", lambda i: f"{seed_no_trailing}/page/{i}/"))

    headers = {
        "Referer": "https://www.shl.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    for name, make_url in strategies:
        no_new_streak = 0
        local: Set[str] = set()

        for i in range(0, max_pages):
            url = canonicalize(make_url(i))
            if not is_allowed(url):
                continue

            try:
                html = fetch_text_with_host_fallback(session, url, cfg, extra_headers=headers)
            except Exception:
                # If a strategy is invalid, it will often fail early. Don’t kill the whole run.
                if i <= 2:
                    break
                no_new_streak += 1
                if no_new_streak >= stop_after_no_new:
                    break
                continue

            found = extract_a_tag_item_links(url, html) | extract_item_urls_from_embedded_json(url, html)
            new = found - local

            if new:
                local |= new
                no_new_streak = 0
            else:
                no_new_streak += 1
                if no_new_streak >= stop_after_no_new:
                    break

        if local:
            logger.info("Listing pagination strategy=%s found=%d", name, len(local))
            discovered |= local

    return discovered


# -----------------------------
# Sitemap parsing (.xml.gz) - best effort
# -----------------------------
def extract_sitemaps_from_robots(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                out.append(sm)
    return out


def parse_sitemap_locs(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)

        def ends(tag: str, suffix: str) -> bool:
            return tag.lower().endswith(suffix)

        locs: List[str] = []
        if ends(root.tag, "sitemapindex"):
            for sm in root:
                for child in sm:
                    if ends(child.tag, "loc") and child.text:
                        locs.append(child.text.strip())
        elif ends(root.tag, "urlset"):
            for u in root:
                for child in u:
                    if ends(child.tag, "loc") and child.text:
                        locs.append(child.text.strip())
        return locs
    except Exception:
        return [m.group(1).strip() for m in _LOC_RE.finditer(xml_text)]


def fetch_sitemap_text(session, url: str, cfg: FetchConfig) -> str:
    data = fetch_bytes_with_host_fallback(
        session,
        url,
        cfg,
        extra_headers={"Accept": "application/xml,text/xml,*/*;q=0.8", "Referer": "https://www.shl.com/"},
    )
    if url.lower().endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def discover_items_via_sitemaps(session, cfg: FetchConfig, state: Dict[str, Set[str]], sitemap_max: int) -> Set[str]:
    seed_sitemaps = [PROFILE.root_sitemap]
    try:
        robots = fetch_text_with_host_fallback(session, PROFILE.robots, cfg, extra_headers={"Referer": "https://www.shl.com/"})
        seed_sitemaps.extend(extract_sitemaps_from_robots(robots))
    except Exception as e:
        logger.warning("robots.txt fetch failed (%s). Using root sitemap only.", e)

    queue: List[str] = []
    for sm in seed_sitemaps:
        sm = canonicalize(sm)
        if is_allowed(sm) and is_sitemap(sm):
            queue.append(sm)

    discovered: Set[str] = set()
    visited = 0

    while queue and visited < sitemap_max:
        sm_url = canonicalize(queue.pop(0))
        if sm_url in state["visited_sitemaps"]:
            continue
        state["visited_sitemaps"].add(sm_url)
        visited += 1

        try:
            xml_text = fetch_sitemap_text(session, sm_url, cfg)
            locs = parse_sitemap_locs(xml_text)
            if not locs:
                continue
        except Exception:
            continue

        for loc in locs:
            loc = canonicalize(loc)
            if not is_allowed(loc):
                continue
            if is_sitemap(loc):
                queue.append(loc)
            else:
                if looks_like_item(loc):
                    discovered.add(loc)

    return discovered


# -----------------------------
# Playwright discovery (click load-more + scroll)
# -----------------------------
_ITEM_URL_RE = re.compile(
    r"https?://(?:www\.)?shl\.com/[^\"'<> \n\r\t]*product-catalog/view/[^\"'<> \n\r\t]*",
    re.IGNORECASE,
)


def discover_with_playwright(seeds: List[str], headless: bool, max_pages: int, load_more_clicks: int) -> Set[str]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && python -m playwright install chromium") from e

    discovered: Set[str] = set()

    def harvest_text(text: str) -> None:
        if not text:
            return
        for m in _ITEM_URL_RE.finditer(text):
            u = canonicalize(m.group(0))
            if is_allowed(u) and looks_like_item(u):
                discovered.add(u)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )

        # Speed: block heavy assets
        def route_handler(route):
            r = route.request
            if r.resource_type in ("image", "font", "media"):
                route.abort()
            else:
                route.continue_()

        context.route("**/*", route_handler)

        page = context.new_page()

        def on_response(resp):
            try:
                url = resp.url
                if not is_allowed(url):
                    return
                ct = (resp.headers.get("content-type") or "").lower()
                if "json" in ct or "text" in ct or "javascript" in ct or "html" in ct:
                    # Avoid huge payloads
                    clen = int(resp.headers.get("content-length", "0") or "0")
                    if clen and clen > 4_000_000:
                        return
                    body = resp.text()
                    harvest_text(body)
            except Exception:
                return

        page.on("response", on_response)

        for seed in seeds[:max_pages]:
            seed = canonicalize(seed)
            if not is_allowed(seed):
                continue

            try:
                page.goto(seed, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_timeout(2000)

                # Try to reveal more items: click "Load more / Show more / Next" a bunch of times
                for _ in range(load_more_clicks):
                    clicked = False

                    # Buttons by text
                    for label in ("Load more", "Show more", "More", "Next"):
                        loc = page.locator(f"button:has-text('{label}')")
                        if loc.count() > 0:
                            try:
                                loc.first.click(timeout=2000)
                                clicked = True
                                break
                            except Exception:
                                pass

                    # Links by text (sometimes pagination is <a>)
                    if not clicked:
                        for label in ("Next", "More"):
                            loc = page.locator(f"a:has-text('{label}')")
                            if loc.count() > 0:
                                try:
                                    loc.first.click(timeout=2000)
                                    clicked = True
                                    break
                                except Exception:
                                    pass

                    # Always scroll (supports infinite scroll)
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(900)

                    # If nothing clickable and scroll didn’t change anything for a while, break
                    if not clicked:
                        # small extra wait for lazy-load
                        page.wait_for_timeout(700)

                # Now collect all visible links
                hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
                for h in hrefs:
                    u = canonicalize(str(h))
                    if is_allowed(u) and looks_like_item(u):
                        discovered.add(u)

                # Also scan full HTML after interactions
                harvest_text(page.content())

            except Exception as e:
                logger.warning("Playwright seed failed: %s (%s)", seed, e)

        browser.close()

    return discovered


# -----------------------------
# Seed discovery (backup + listing probe)
# -----------------------------
def discover_from_seeds(session, cfg: FetchConfig, seeds: List[str], listing_max_pages: int) -> Set[str]:
    discovered: Set[str] = set()
    for seed in seeds:
        seed = canonicalize(seed)
        if not is_allowed(seed):
            continue

        # 1) Seed page itself
        try:
            html = fetch_text_with_host_fallback(session, seed, cfg, extra_headers={"Referer": "https://www.shl.com/"})
            discovered |= extract_a_tag_item_links(seed, html)
            discovered |= extract_item_urls_from_embedded_json(seed, html)
        except Exception as e:
            logger.warning("Seed fetch failed: %s (%s)", seed, e)

        # 2) Listing pagination probe (IMPORTANT)
        try:
            discovered |= discover_items_via_listing_pagination(session, cfg, seed, max_pages=listing_max_pages)
        except Exception as e:
            logger.warning("Listing probe failed: %s (%s)", seed, e)

    return discovered


# -----------------------------
# Main crawl
# -----------------------------
def run_crawl(
    out_jsonl: str,
    state_file: str,
    user_agent: str,
    extra_seeds: List[str],
    sitemap_max: int,
    listing_max_pages: int,
    use_playwright: bool,
    pw_headless: bool,
    pw_max_pages: int,
    pw_load_more_clicks: int,
) -> None:
    cfg_sitemap = FetchConfig(timeout_s=90, min_delay_s=0.25, max_delay_s=0.7, max_retries=2, backoff_factor=0.8)
    cfg_seed = FetchConfig(timeout_s=90, min_delay_s=0.35, max_delay_s=0.9, max_retries=2, backoff_factor=0.8)
    cfg_item = FetchConfig(timeout_s=45, min_delay_s=0.35, max_delay_s=0.9, max_retries=2, backoff_factor=0.6)

    session = build_session(user_agent=user_agent, cfg=cfg_seed)

    state = load_state(state_file)

    existing = load_existing_urls_from_jsonl(out_jsonl)
    if existing:
        state["parsed_items"] |= existing

    seeds = [canonicalize(s) for s in (PROFILE.catalog_seeds + list(extra_seeds))]

    logger.info("Discovery: sitemap traversal")
    items_from_sitemaps = discover_items_via_sitemaps(session, cfg_sitemap, state, sitemap_max=sitemap_max)
    logger.info("Sitemap discovered item URLs: %d", len(items_from_sitemaps))

    logger.info("Discovery: seed pages + listing pagination probe")
    items_from_seeds = discover_from_seeds(session, cfg_seed, seeds, listing_max_pages=listing_max_pages)
    logger.info("Seed+listing discovered item URLs: %d", len(items_from_seeds))

    items_from_pw: Set[str] = set()
    if use_playwright:
        logger.info("Discovery: Playwright (click load-more + scroll)")
        items_from_pw = discover_with_playwright(
            seeds=seeds,
            headless=pw_headless,
            max_pages=pw_max_pages,
            load_more_clicks=pw_load_more_clicks,
        )
        logger.info("Playwright discovered item URLs: %d", len(items_from_pw))

    discovered_items = set(items_from_sitemaps) | set(items_from_seeds) | set(items_from_pw)
    logger.info("TOTAL discovered item URLs: %d", len(discovered_items))

    save_state(state_file, state)

    parsed_now = 0
    for item_url in tqdm(sorted(discovered_items), desc="Parsing item pages"):
        if item_url in state["parsed_items"]:
            continue

        try:
            html = fetch_text_with_host_fallback(session, item_url, cfg_item, extra_headers={"Referer": "https://www.shl.com/"})
            title, main_text, meta = parse_item(html, item_url)
            fields = map_fields(title=title, main_text=main_text, meta=meta)

            if not (fields["name"] or "").strip():
                continue

            item = CatalogItem(
                url=item_url,
                name=fields["name"],
                description=fields["description"],
                duration=fields["duration"],
                adaptive_support=fields["adaptive_support"],
                remote_support=fields["remote_support"],
                test_type=fields["test_type"],
                source="shl",
            )

            append_jsonl(out_jsonl, item.model_dump(mode="json"))
            state["parsed_items"].add(item_url)
            parsed_now += 1

        except Exception as e:
            logger.warning("Item parse failed: %s (%s)", item_url, e)

        if parsed_now > 0 and parsed_now % 50 == 0:
            save_state(state_file, state)

    save_state(state_file, state)
    logger.info("DONE. Parsed items written this run=%d -> %s", parsed_now, out_jsonl)
    logger.info("TOTAL parsed items in state=%d", len(state["parsed_items"]))


def main():
    parser = argparse.ArgumentParser(description="Crawl SHL product catalog into JSONL")
    parser.add_argument("--out", default="data/catalog.jsonl")
    parser.add_argument("--state", default="data/state.json")
    parser.add_argument(
        "--ua",
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    )
    parser.add_argument("--seed", action="append", default=[], help="Extra seed URLs (repeatable)")
    parser.add_argument("--sitemap-max", type=int, default=1200)
    parser.add_argument("--listing-max-pages", type=int, default=120, help="Pagination probe pages per seed")

    # Playwright options
    parser.add_argument("--use-playwright", action="store_true", help="Use Playwright for discovery (recommended)")
    parser.add_argument("--pw-headless", action="store_true", help="Run Playwright headless")
    parser.add_argument("--pw-max-pages", type=int, default=2, help="How many seed pages to open with Playwright")
    parser.add_argument("--pw-load-more-clicks", type=int, default=60, help="How many times to click/scroll to reveal more items")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    run_crawl(
        out_jsonl=args.out,
        state_file=args.state,
        user_agent=args.ua,
        extra_seeds=args.seed,
        sitemap_max=args.sitemap_max,
        listing_max_pages=args.listing_max_pages,
        use_playwright=args.use_playwright,
        pw_headless=args.pw_headless,
        pw_max_pages=args.pw_max_pages,
        pw_load_more_clicks=args.pw_load_more_clicks,
    )


if __name__ == "__main__":
    main()

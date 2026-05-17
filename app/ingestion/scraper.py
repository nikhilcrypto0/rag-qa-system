import logging
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "SEMO-RAG-Bot/1.0 (university-research-project)"}
REQUEST_TIMEOUT = 15


def _is_same_domain(url: str, base_domain: str) -> bool:
    try:
        return urlparse(url).netloc.endswith(base_domain)
    except Exception:
        return False


def _extract_text(html: str, url: str) -> Tuple[str, str]:
    """Return (title, clean_text) from raw HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else ""
    return title, text


def _collect_links(html: str, base_url: str, allowed_domain: str) -> List[str]:
    """Extract all same-domain links from a page."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        # Only http/https, same domain, no fragments, no query params for crawl
        if (
            parsed.scheme in ("http", "https")
            and _is_same_domain(href, allowed_domain)
            and not parsed.fragment
        ):
            links.append(href.split("#")[0])
    return list(set(links))


def scrape_url(url: str, allowed_domain: str = "") -> List[Document]:
    """Scrape a single URL. Raises on network error."""
    if allowed_domain and not _is_same_domain(url, allowed_domain):
        raise ValueError(f"URL {url} is outside allowed domain {allowed_domain}")

    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    title, text = _extract_text(resp.text, url)
    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={"source": url, "title": title, "content_type": "web_scraped"},
    )]


def scrape_recursive(
    start_url: str,
    allowed_domain: str,
    max_depth: int = 2,
    max_pages: int = 50,
) -> Tuple[List[Document], List[dict]]:
    """
    Crawl start_url and follow internal links up to max_depth levels.
    Returns (documents, errors).
    """
    visited: set = set()
    queue = [(start_url, 0)]  # (url, depth)
    all_docs: List[Document] = []
    errors: List[dict] = []

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            title, text = _extract_text(resp.text, url)

            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": url, "title": title, "content_type": "web_scraped"},
                ))

            # Follow links if not at max depth
            if depth < max_depth:
                links = _collect_links(resp.text, url, allowed_domain)
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", url, exc)
            errors.append({"url": url, "error": str(exc)})

    logger.info("Scraped %d pages, %d errors", len(all_docs), len(errors))
    return all_docs, errors

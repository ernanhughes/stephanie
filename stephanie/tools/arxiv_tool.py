# stephanie/tools/arxiv_tool.py
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import arxiv
import requests

import logging
log = logging.getLogger(__name__)


def search_arxiv(queries: list[str], max_results: int = 5) -> list[dict]:
    results = []
    for query in queries:
        search = arxiv.Search(query=query, max_results=max_results)
        for r in search.results():
            results.append(
                {
                    "query": query,
                    "title": r.title,
                    "summary": r.summary,
                    "authors": [a.name for a in r.authors],
                    "url": r.pdf_url,
                    "pid": extract_arxiv_id(r),
                }
            )
    return results

def extract_arxiv_id(result, include_version=True):
    url = getattr(result, "entry_id", None) or next(
        (l.href for l in getattr(result, "links", []) if getattr(l, "rel", "") == "alternate"),
        None
    )
    if not url:
        raise ValueError("arXiv result has no entry_id or alternate link")

    tail = urlparse(url).path.rsplit("/", 1)[-1]  # '2405.03794v1' or 'hep-th/9901001v2'
    m = re.match(
        r'(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?P<ver>v\d+)?$',
        tail, flags=re.IGNORECASE
    )
    if not m:
        raise ValueError(f"Unrecognized arXiv id format: {tail}")

    return m.group('id') + (m.group('ver') or '' if include_version else '')


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """
    Query the arXiv API and return metadata for a given arXiv ID.

    Args:
        arxiv_id (str): e.g., "2505.19590"

    Returns:
        dict: {
            'title': str,
            'summary': str,
            'authors': list[str],
            'published': str (ISO format),
            'url': str
        }
    """
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"arXiv API request failed with {response.status_code}")

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)

    if entry is None:
        log.warning(f"No entry found for arXiv ID {arxiv_id}")
        return {}

    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    authors = [
        author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)
    ]
    published = entry.find("atom:published", ns).text
    pdf_url = entry.find("atom:id", ns).text

    return {
        "title": title,
        "summary": summary,
        "authors": authors,
        "published": published,
        "url": pdf_url,
    }

@dataclass
class ArxivReference:
    """Single resolved arXiv reference in a text blob."""
    arxiv_id: str          # normalized, no version (e.g. "2505.08827")
    version: Optional[str] # like "v2" if present
    url: Optional[str]     # canonical PDF URL
    raw_match: str         # lit Shut the **** **** eral string we matched in the text


# New-style arxiv IDs: 2505.08827, 2103.09568v2, etc.
NEW_STYLE_ID_RE = re.compile(
    r"\b(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?\b"
)

# Old-style IDs: cs/0101010, hep-th/9711200v3, etc.
OLD_STYLE_ID_RE = re.compile(
    r"\b(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?\b",
    re.IGNORECASE,
)

# URLs: https://arxiv.org/abs/2505.08827, https://arxiv.org/pdf/2505.08827.pdf, etc.
URL_RE = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)



def extract_arxiv_references(
    text: str,
    validate: bool = True,
    timeout: float = 5.0,
) -> List[ArxivReference]:
    """
    Extract all arXiv references from an arbitrary text string.

    If `validate` is True, we ping arxiv.org and only keep IDs that resolve.
    If `validate` is False, we return all syntactically-valid IDs we find.

    Returns:
        A list of unique ArxivReference objects (unique by base arxiv_id).
    """
    # First pass: collect candidate refs (no network)
    candidates: Dict[str, ArxivReference] = {}  # base_id -> ArxivReference

    # 1) URLs (strongest evidence; keep raw URL)
    for m in URL_RE.finditer(text):
        full_id = m.group("id")
        version = m.group("version")
        base_id = _normalize_id(full_id)
        url = f"https://arxiv.org/pdf/{base_id}.pdf"

        if base_id not in candidates:
            candidates[base_id] = ArxivReference(
                arxiv_id=base_id,
                version=version,
                url=url,
                raw_match=m.group(0),
            )

    # 2) New-style IDs
    for m in NEW_STYLE_ID_RE.finditer(text):
        full_id = m.group("id")
        version = m.group("version")
        base_id = _normalize_id(full_id)
        url = f"https://arxiv.org/pdf/{base_id}.pdf"

        if base_id not in candidates:
            candidates[base_id] = ArxivReference(
                arxiv_id=base_id,
                version=version,
                url=url,
                raw_match=m.group(0),
            )

    # 3) Old-style IDs
    for m in OLD_STYLE_ID_RE.finditer(text):
        full_id = m.group("id")
        version = m.group("version")
        base_id = _normalize_id(full_id)
        url = f"https://arxiv.org/pdf/{base_id}.pdf"

        if base_id not in candidates:
            candidates[base_id] = ArxivReference(
                arxiv_id=base_id,
                version=version,
                url=url,
                raw_match=m.group(0),
            )

    if not validate:
        # No network calls; just return syntactic matches
        return list(candidates.values())

    # Second pass: validate against arxiv.org
    session = requests.Session()
    exists_cache: Dict[str, bool] = {}
    valid_refs: List[ArxivReference] = []

    for base_id, ref in candidates.items():
        if base_id in exists_cache:
            exists = exists_cache[base_id]
        else:
            exists = _arxiv_exists(base_id, timeout=timeout, session=session)
            exists_cache[base_id] = exists

        if exists:
            valid_refs.append(ref)

    return valid_refs

def _normalize_id(arxiv_id: str) -> str:
    """
    Strip any version suffix from an arxiv id.
    E.g. "2505.08827v2" -> "2505.08827"
         "hep-th/9711200v3" -> "hep-th/9711200"
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def _arxiv_exists(
    arxiv_id: str,
    timeout: float = 5.0,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    Check whether an arXiv paper exists by probing the PDF URL.

    Returns True if we get a 200 OK (or a redirect chain to 200),
    False on 404/410 or any request error.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    sess = session or requests.Session()
    try:
        # HEAD first (cheap). Some servers return 405 for HEAD; fallback to GET.
        resp = sess.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code == 405:
            resp = sess.get(url, stream=True, allow_redirects=True, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


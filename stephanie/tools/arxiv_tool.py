# stephanie/tools/arxiv_tool.py
import xml.etree.ElementTree as ET

import arxiv
import requests


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
                }
            )
    return results


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
        raise ValueError(f"No entry found for arXiv ID {arxiv_id}")

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

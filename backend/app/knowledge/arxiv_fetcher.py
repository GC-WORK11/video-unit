"""ArXiv knowledge fetcher — downloads and chunks relevant physics/engineering papers."""
import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timedelta

import arxiv

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)

ARXIV_CACHE = DATA_DIR / "arxiv_cache"
ARXIV_CACHE.mkdir(parents=True, exist_ok=True)

# Search queries for AETHER's knowledge base
ARXIV_QUERIES = [
    # Mechanical systems & dynamics
    {"query": "belt drive mechanical vibration dynamics", "category": "mechanics", "max_results": 30},
    {"query": "gantry robot precision control", "category": "robotics", "max_results": 20},
    {"query": "prismatic joint mechanism dynamics", "category": "mechanics", "max_results": 20},
    {"query": "pulley system mechanical advantage", "category": "mechanics", "max_results": 20},
    # Control theory
    {"query": "model predictive control robotics", "category": "control", "max_results": 25},
    {"query": "LQR optimal control mechanical systems", "category": "control", "max_results": 20},
    {"query": "MPC friction compensation", "category": "control", "max_results": 15},
    # Computer vision for perception
    {"query": "video object segmentation deep learning", "category": "vision", "max_results": 20},
    {"query": "point tracking optical flow", "category": "vision", "max_results": 15},
    {"query": "monocular depth estimation", "category": "vision", "max_results": 20},
    # Materials & friction
    {"query": "belt friction mechanical contact", "category": "materials", "max_results": 15},
    {"query": "Coulomb friction dynamic systems", "category": "physics", "max_results": 15},
    # Dynamics
    {"query": "rigid body dynamics Lagrangian Hamiltonian", "category": "physics", "max_results": 20},
    {"query": "vibration analysis structural dynamics", "category": "physics", "max_results": 20},
    {"query": "motor torque control servo", "category": "robotics", "max_results": 15},
]


def _chunk_text(text: str, chunk_size: int = 350, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        words = len(sentence.split())
        if current_len + words > chunk_size and current:
            chunks.append(' '.join(current))
            # Keep overlap
            overlap_sentences = current[-2:] if len(current) >= 2 else current[-1:]
            current = overlap_sentences + [sentence]
            current_len = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            current_len += words

    if current:
        chunks.append(' '.join(current))

    return chunks


def _paper_to_chunks(paper: dict) -> list[dict]:
    """Convert a paper into embedding-ready chunks."""
    chunks = []

    # Title chunk
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    authors = paper.get("authors", ["Unknown"])[0]
    published = paper.get("published", "")
    url = paper.get("url", "")

    # Full text for chunking
    full_text = f"{title}. {abstract}"

    text_chunks = _chunk_text(full_text)
    for i, chunk in enumerate(text_chunks):
        chunk_id = hashlib.sha256(f"{url}_{i}_{chunk[:50]}".encode()).hexdigest()[:16]
        chunks.append({
            "id": chunk_id,
            "text": chunk.strip(),
            "title": title,
            "source": f"arXiv:{paper.get('arxiv_id', '')}",
            "category": paper.get("category", "general"),
            "url": url,
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            "authors": authors,
            "published": published,
        })

    return chunks


def fetch_arxiv_papers(queries: list[dict] | None = None) -> list[dict]:
    """Fetch papers from ArXiv based on queries.

    Returns list of paper dicts with title, abstract, authors, url, arxiv_id.
    """
    if queries is None:
        queries = ARXIV_QUERIES

    all_papers = []
    seen_ids = set()

    client = arxiv.Client()

    for q in queries:
        query_str = q["query"]
        category = q["category"]
        max_results = q.get("max_results", 20)

        log.info(f"Searching ArXiv: '{query_str}' ({max_results} results)")

        try:
            search = arxiv.Search(
                query=query_str,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            for result in client.results(search):
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)

                paper = {
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title or "Untitled",
                    "abstract": result.summary or "",
                    "authors": [a.name for a in (result.authors or [])],
                    "published": str(result.published.date()) if result.published else "",
                    "url": result.entry_id,
                    "category": category,
                    "comment": result.comment or "",
                    "doi": result.doi or "",
                }
                all_papers.append(paper)

            # Be respectful to arxiv API
            time.sleep(0.5)

        except Exception as e:
            log.warning(f"ArXiv search failed for '{query_str}': {e}")
            continue

    log.info(f"Fetched {len(all_papers)} papers from ArXiv")
    return all_papers


def ingest_papers(papers: list[dict]) -> list[dict]:
    """Ingest papers into the knowledge base.

    Returns list of all chunks added.
    """
    from app.knowledge.chromadb import add_documents

    all_chunks = []
    for paper in papers:
        chunks = _paper_to_chunks(paper)
        all_chunks.extend(chunks)

    if all_chunks:
        add_documents(all_chunks)
        log.info(f"Ingested {len(all_chunks)} chunks from {len(papers)} papers")

    return all_chunks


def quick_ingest(query: str, category: str = "general", max_results: int = 10) -> dict:
    """One-shot: fetch, chunk, and ingest papers for a query.

    Returns stats about the ingestion.
    """
    papers = fetch_arxiv_papers([{"query": query, "category": category, "max_results": max_results}])
    chunks = ingest_papers(papers)
    return {
        "papers_fetched": len(papers),
        "chunks_added": len(chunks),
        "query": query,
    }

"""Retrieval layer: discovers ChromaDB backends, runs semantic search, builds LLM context."""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Maps the many ways a mission can be written to the value stored in chunk metadata.
MISSION_ALIASES = {
    "apollo_11": "apollo_11", "apollo11": "apollo_11", "apollo 11": "apollo_11",
    "apollo_13": "apollo_13", "apollo13": "apollo_13", "apollo 13": "apollo_13",
    "challenger": "challenger", "sts-51l": "challenger", "sts_51l": "challenger",
}

# How each mission is written in user questions.
MISSION_PATTERNS = {
    "apollo_11": re.compile(r"\bapollo[\s_-]*(?:11|xi)\b", re.IGNORECASE),
    "apollo_13": re.compile(r"\bapollo[\s_-]*(?:13|xiii)\b", re.IGNORECASE),
    "challenger": re.compile(r"\b(?:space\s+shuttle\s+)?challenger(?:'s)?\b|\bsts[\s_-]*51[\s_-]*l\b",
                             re.IGNORECASE),
}

MAX_DOC_CHARS = 1500  # per-chunk cap inside the LLM context


def _client_settings() -> Settings:
    return Settings(anonymized_telemetry=False)


def _ensure_embedding_key() -> None:
    """Collections persist an OpenAI embedding function that reads CHROMA_OPENAI_API_KEY.

    Fall back to OPENAI_API_KEY so the collection can embed queries either way.
    """
    if not os.environ.get("CHROMA_OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]


def normalize_mission(mission: Optional[str]) -> Optional[str]:
    """Return the metadata value for a mission name, or None for 'all'/unknown values."""
    if not mission:
        return None
    key = mission.strip().lower()
    if key in ("all", "all missions", "any", "none", ""):
        return None
    return MISSION_ALIASES.get(key, key.replace(" ", "_"))


def detect_missions(query: str) -> List[str]:
    """Return the missions named in a question, e.g. ['apollo_13']."""
    return [mission for mission, pattern in MISSION_PATTERNS.items() if pattern.search(query or "")]


def prepare_query(query: str, mission_filter: Optional[str] = None,
                  auto_detect_mission: bool = True):
    """Choose the mission filter and the text to embed for a question.

    - With no filter, a question naming exactly one mission gets that mission's filter.
    - With a filter, the mission's name is removed from the text to embed. Almost every
      chunk of a mission repeats its name, so the name only adds noise to the similarity
      ranking once results are restricted to that mission.

    Returns:
        Tuple (search_text, mission or None).
    """
    mission = normalize_mission(mission_filter)
    if mission is None and auto_detect_mission:
        mentioned = detect_missions(query)
        if len(mentioned) == 1:
            mission = mentioned[0]

    search_text = query
    if mission in MISSION_PATTERNS:
        stripped = MISSION_PATTERNS[mission].sub(" ", query)
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,.;:-")
        if len(re.findall(r"\w+", stripped)) >= 4:  # keep enough words to be meaningful
            search_text = stripped

    return search_text, mission


def discover_chroma_backends(search_dir: str = ".") -> Dict[str, Dict[str, str]]:
    """Discover ChromaDB directories (names containing 'chroma') and their collections.

    Returns:
        Mapping of "<directory>::<collection>" to info dicts with keys
        directory, collection_name, display_name and document_count.
    """
    backends: Dict[str, Dict[str, str]] = {}
    base = Path(search_dir)

    chroma_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and "chroma" in d.name.lower() and not d.name.startswith(".")
    )

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir), settings=_client_settings())
            for collection in client.list_collections():
                # chromadb>=0.6 returns Collection objects; older versions returned names.
                name = getattr(collection, "name", collection)
                try:
                    count = client.get_collection(name).count()
                except Exception:
                    count = "unknown"

                backends[f"{chroma_dir.name}::{name}"] = {
                    "directory": str(chroma_dir),
                    "collection_name": name,
                    "display_name": f"{name} ({chroma_dir.name}, {count} chunks)",
                    "document_count": str(count),
                }
        except Exception as e:
            logger.warning("Could not open ChromaDB directory %s: %s", chroma_dir, e)
            error_text = str(e)
            backends[f"{chroma_dir.name}::error"] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (unavailable: {error_text[:50]}"
                                f"{'...' if len(error_text) > 50 else ''})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Open a persisted collection.

    Returns:
        Tuple (collection, success, error_message).
    """
    if not collection_name:
        return None, False, f"No usable collection found in {chroma_dir}"

    try:
        _ensure_embedding_key()
        client = chromadb.PersistentClient(path=chroma_dir, settings=_client_settings())
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def _deduplicate_results(results: Dict, n_results: int) -> Dict:
    """Drop repeated snippets and keep the n_results best hits sorted by distance."""
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0] or [{}] * len(documents)
    distances = (results.get("distances") or [[]])[0] or [0.0] * len(documents)
    ids = (results.get("ids") or [[]])[0] or [""] * len(documents)

    ranked = sorted(zip(documents, metadatas, distances, ids), key=lambda item: item[2])

    seen = set()
    kept = []
    for doc, meta, dist, doc_id in ranked:
        fingerprint = " ".join((doc or "").lower().split())
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append((doc, meta or {}, dist, doc_id))
        if len(kept) >= n_results:
            break

    return {
        "ids": [[k[3] for k in kept]],
        "documents": [[k[0] for k in kept]],
        "metadatas": [[k[1] for k in kept]],
        "distances": [[k[2] for k in kept]],
    }


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None,
                       auto_detect_mission: bool = True) -> Optional[Dict]:
    """Semantic search over the collection with optional mission filtering.

    The collection's OpenAI embedding function embeds the question, and ChromaDB
    returns the nearest chunks by cosine distance. A few extra candidates are fetched
    so duplicate snippets can be removed while still returning n_results hits.

    The result also holds "search_query" and "mission_filter", showing what was searched.
    """
    if collection is None or not query or not query.strip():
        return None

    n_results = max(1, int(n_results))
    search_text, mission = prepare_query(query, mission_filter, auto_detect_mission)
    where_filter = {"mission": mission} if mission else None

    results = collection.query(
        query_texts=[search_text],
        n_results=n_results * 2,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    deduplicated = _deduplicate_results(results, n_results)
    deduplicated["search_query"] = search_text
    deduplicated["mission_filter"] = mission
    return deduplicated


def _pretty(value: str) -> str:
    return str(value).replace("_", " ").title()


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved chunks into one context string with numbered source headers."""
    if not documents:
        return ""

    metadatas = metadatas or [{}] * len(documents)
    context_parts = ["=== Retrieved NASA archive context ==="]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}
        mission = _pretty(metadata.get("mission", "unknown mission"))
        source = metadata.get("source", "unknown source")
        category = _pretty(metadata.get("document_category", "general"))

        header = f"[Source {i}] Mission: {mission} | Document: {source} | Category: {category}"
        if "chunk_index" in metadata:
            header += f" | Chunk: {metadata['chunk_index']}"
        if distances and i - 1 < len(distances) and distances[i - 1] is not None:
            header += f" | Similarity: {1 - distances[i - 1]:.3f}"

        text = (doc or "").strip()
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS].rstrip() + " ..."

        context_parts.append("-" * 60)
        context_parts.append(header)
        context_parts.append(text)

    context_parts.append("=" * 60)
    return "\n".join(context_parts)

import re
from .state import FTOState

_SIGNIFICANT_OVERLAP_THRESHOLD = 0.2
_PATENT_ID_RE = re.compile(r"\b[A-Z]{2}\.\d+\.[A-Z]\d*\b")


def sort_assessments_by_overlap(assessments: list) -> list:
    return sorted(assessments, key=lambda a: a.get("overlap_score") or 0, reverse=True)


def find_hallucinated_patent_ids(report: str, allowed_ids: set[str]) -> set[str]:
    mentioned = set(_PATENT_ID_RE.findall(report or ""))
    return mentioned - allowed_ids

_RELEVANCE_ABSTRACT_MAX = 500
_RELEVANCE_CLAIM_EXCERPT_MAX = 400
_RELEVANCE_TOTAL_MAX = 1000


def _extract_abstract(text_for_vector_db: str) -> str:
    if not text_for_vector_db:
        return ""
    if "Abstract:" in text_for_vector_db:
        return text_for_vector_db.split("Abstract:", 1)[1].strip()
    lines = text_for_vector_db.strip().splitlines()
    if lines and lines[0].startswith("Title:"):
        return "\n".join(lines[1:]).strip()
    return text_for_vector_db.strip()


def build_relevance_context(patent: dict) -> str:
    title = patent.get("title", "").strip()
    abstract_raw = _extract_abstract(patent.get("text_for_vector_db", ""))
    abstract = " ".join(abstract_raw.split())[:_RELEVANCE_ABSTRACT_MAX]

    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract: {abstract}")

    if patent.get("context_type") == "claim":
        claims_text = (patent.get("claims_text") or "").strip()
        if claims_text:
           excerpt = " ".join(claims_text.split())[:_RELEVANCE_CLAIM_EXCERPT_MAX]
           parts.append(f"Claims (excerpt): {excerpt}")

    return "\n".join(parts)[:_RELEVANCE_TOTAL_MAX]


def sanitize_epo_query(raw: str) -> str:
    cleaned = re.sub(r'[`"\'\(\)\[\]\\*?]', '', raw)
    cleaned = re.sub(r'\b(AND|OR|NOT|CQL)\b', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# queries 1-2 get picked first in the merge
CORE_QUERY_INDICES = (0, 1)
CORE_RESERVE = 10


def _round_robin_pick(
    query_results: list[list[str]],
    indices: range | tuple[int, ...],
    seen: set[str],
    top_ids: list[str],
    n: int,
    max_rank: int,
) -> None:
    for rank in range(max_rank):
        if len(top_ids) >= n:
            return
        for q_idx in indices:
            if len(top_ids) >= n:
                return
            if q_idx < len(query_results) and rank < len(query_results[q_idx]):
                pid = query_results[q_idx][rank]
                if pid not in seen:
                    seen.add(pid)
                    top_ids.append(pid)


def merge_top_patent_ids(query_results: list[list[str]], n: int = 20) -> list[str]:
    top_ids: list[str] = []
    seen: set[str] = set()

    _round_robin_pick(
        query_results, CORE_QUERY_INDICES, seen, top_ids, CORE_RESERVE, max_rank=8
    )
    _round_robin_pick(
        query_results, range(len(query_results)), seen, top_ids, n, max_rank=8
    )

    return top_ids


def route_after_quality_check(state: FTOState) -> str:
    score = state["retrieval_quality_score"]
    retries = state["retry_count"]
    if score == 0.0 and retries >= 2:
        return "no_results"
    if score < 0.4 and retries < 2:
        return "retry"
    return "chromadb_seeder"

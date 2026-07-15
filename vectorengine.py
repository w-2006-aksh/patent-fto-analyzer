import re
import chromadb

# max chars when dumping raw claims as one blob
_RAW_TEXT_MAX_CHARS = 10_000

_sessions: dict[str, tuple] = {}


def create_ephemeral_collection(session_id: str):
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="patent_claims")
    _sessions[session_id] = (client, collection)
    return collection


def get_ephemeral_collection(session_id: str):
    entry = _sessions.get(session_id)
    return entry[1] if entry else None


def drop_ephemeral_session(session_id: str):
    _sessions.pop(session_id, None)


def load_claims_to_db(collection, patents: list):
    # split numbered claims if possible, else raw text, else abstract
    if not patents:
        print("no patents to load")
        return

    documents = []
    metadatas = []
    ids = []

    for patent in patents:
        patent_id = patent["id"]
        patent_title = patent.get("title", "")
        claims_text = patent.get("claims_text", "").strip()
        context_type = patent.get("context_type", "claim")

        if context_type == "abstract_fallback":
            abstract_text = claims_text or patent.get("text_for_vector_db", "")
            if not abstract_text:
                print(f"{patent_id}: no text, skipping")
                continue
            entry_id = f"{patent_id}_abstract"
            print(f"storing abstract for {patent_id}")
            documents.append(abstract_text)
            metadatas.append({
                "patent_id":    patent_id,
                "patent_title": patent_title,
                "claim_number": 0,
                "context_type": "abstract_fallback",
            })
            ids.append(entry_id)
            continue

        if claims_text:
            # split on "1. ", "2. " at line start
            parts  = re.split(r"(?m)(?=^\d+\.\s)", claims_text)
            chunks = [p.strip() for p in parts if p.strip()]

            parsed: list[tuple[int, str]] = []
            for chunk in chunks:
                m = re.match(r"^(\d+)\.\s*(.*)", chunk, re.DOTALL)
                if m:
                    parsed.append((int(m.group(1)), m.group(2).strip()))
                else:
                    print(f"{patent_id}: could not parse claim chunk")

            if parsed:
                for claim_number, claim_body in parsed:
                    entry_id = f"{patent_id}_c{claim_number}"
                    print(f"storing claim {claim_number} for {patent_id}")
                    documents.append(claim_body)
                    metadatas.append({
                        "patent_id":    patent_id,
                        "patent_title": patent_title,
                        "claim_number": claim_number,
                        "context_type": "claim",
                    })
                    ids.append(entry_id)
                continue

            # regex didn't parse. store whole block
            raw = claims_text
            if len(raw) > _RAW_TEXT_MAX_CHARS:
                raw = raw[:_RAW_TEXT_MAX_CHARS]
                print(f"{patent_id}: raw claims truncated")
            entry_id = f"{patent_id}_raw"
            print(f"storing raw claims for {patent_id}")
            documents.append(raw)
            metadatas.append({
                "patent_id":    patent_id,
                "patent_title": patent_title,
                "claim_number": 1,
                "context_type": "unformatted_claims",
            })
            ids.append(entry_id)
            continue

        abstract_text = patent.get("text_for_vector_db", "")
        if not abstract_text:
            print(f"{patent_id}: no claims or abstract, skipping")
            continue
        entry_id = f"{patent_id}_abstract"
        print(f"storing abstract for {patent_id}")
        documents.append(abstract_text)
        metadatas.append({
            "patent_id":    patent_id,
            "patent_title": patent_title,
            "claim_number": 0,
            "context_type": "abstract_fallback",
        })
        ids.append(entry_id)

    if not ids:
        print("no claims to load")
        return

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"loaded {len(ids)} claims into chromadb")


def _hit_from_meta(doc: str, meta: dict) -> dict:
    return {
        "patent_id":    meta["patent_id"],
        "patent_title": meta["patent_title"],
        "claim_number": meta["claim_number"],
        "context_type": meta.get("context_type", "claim"),
        "claim_text":   doc,
    }


def retrieve_top_claims_per_patent(
    collection, user_idea: str, patent_ids: list, n_per_patent: int = 2
) -> list:
    # top n claims per patent 
    if collection.count() == 0:
        return []

    hits = []
    for pid in patent_ids:
        try:
            results = collection.query(
                query_texts=[user_idea],
                n_results=min(n_per_patent, collection.count()),
                where={"patent_id": pid},
            )
        except Exception as exc:
            print(f"chromadb query failed for {pid}: {exc}")
            continue

        for i, doc in enumerate(results["documents"][0]):
            hits.append(_hit_from_meta(doc, results["metadatas"][0][i]))

    return hits

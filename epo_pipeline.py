import os
import base64
import time
import requests
import json
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
EPO_KEY = os.getenv("EPO_API_KEY")
EPO_SECRET = os.getenv("EPO_API_SECRET")

_DEFAULT_TIMEOUT = 30


def _epo_get(url, headers, params=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        response = requests.get(url, headers=headers, params=params, timeout=_DEFAULT_TIMEOUT)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"epo rate limit, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        return response
    return response


def _epo_post(url, headers, data, max_retries=3):
    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, data=data, timeout=_DEFAULT_TIMEOUT)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"epo rate limit, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        return response
    return response


def get_epo_token():
    auth_url = "https://ops.epo.org/3.2/auth/accesstoken"
    credentials = base64.b64encode(f"{EPO_KEY}:{EPO_SECRET}".encode()).decode()

    response = _epo_post(
        auth_url,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={"grant_type": "client_credentials"}
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_and_get_ids(token, query, count=10):
    response = _epo_get(
        "https://ops.epo.org/3.2/rest-services/published-data/search/biblio",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-OPS-Range": f"1-{count}"
        },
        params={"q": f"ta={query}"}
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()

    try:
        references = (
            response.json()
            ["ops:world-patent-data"]
            ["ops:biblio-search"]
            ["ops:search-result"]
            ["exchange-documents"]
        )
    except (KeyError, ValueError):
        return []

    if isinstance(references, dict):
        references = [references]

    ids = []
    for doc in references:
        try:
            ed = doc["exchange-document"]
            ids.append(f"{ed['@country']}.{ed['@doc-number']}.{ed['@kind']}")
        except (KeyError, TypeError):
            continue
    return ids


def hydrate_patents(token, id_list):
    if not id_list:
        return []

    batch_string = ",".join(id_list)
    print(f"fetching biblio for {len(id_list)} patents")

    try:
        response = _epo_get(
            f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{batch_string}/biblio",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"biblio request failed: {exc}")
        return []

    world_data = response.json().get("ops:world-patent-data", {})

    # mixed batches drop the ops: prefix on exchange-documents
    container = (
        world_data.get("exchange-documents")
        or world_data.get("ops:exchange-documents")
        or {}
    )
    documents = container.get("exchange-document", [])

    if isinstance(documents, dict):
        documents = [documents]

    results = []
    for doc in documents:
        try:
            results.append(_extract_fields(doc))
        except Exception as exc:
            print(f"skipping bad document: {exc}")

    return results


hydrate_and_clean_batch = hydrate_patents


def fetch_patent_claims(token: str, patent_id: str) -> str:
    try:
        response = _epo_get(
            f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{patent_id}/claims",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/xml",
            }
        )

        if response.status_code == 404:
            print(f"{patent_id}: claims not found (404)")
            return ""

        response.raise_for_status()

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            print(f"{patent_id}: xml parse error: {exc}")
            return ""

        # each <claim-text> becomes one line for the numbered-claim regex later
        claims_blocks = root.findall(".//{*}claims")
        en_blocks = [b for b in claims_blocks if (b.get("lang") or "").upper() == "EN"]
        blocks_to_parse = en_blocks if en_blocks else claims_blocks

        if not blocks_to_parse:
            print(f"{patent_id}: no claims in xml")
            return ""

        texts: list[str] = []
        for block in blocks_to_parse:
            for node in block.findall(".//{*}claim-text"):
                text = " ".join(node.itertext()).split()
                if text:
                    texts.append(" ".join(text))

        result = "\n".join(texts)
        print(f"{patent_id}: got {len(texts)} claims")
        return result

    except Exception as exc:
        print(f"{patent_id}: claims error: {exc}")
        return ""



def _extract_fields(doc):
    # needs dotted id like EP.3695720.A1 for claims endpoint
    doc_id = f"{doc.get('@country', '')}.{doc.get('@doc-number', '')}.{doc.get('@kind', '')}"
    biblio = doc.get("bibliographic-data", {})

    title_nodes = biblio.get("invention-title", [])
    title = _pick_english(title_nodes, value_key="$")
    abstract = _pick_english(doc.get("abstract", []), nested_key="p", value_key="$")

    return {
        "id": doc_id,
        "title": title,
        "text_for_vector_db": f"Title: {title}\nAbstract: {abstract}"
    }


def _pick_english(nodes, nested_key=None, value_key="$"):
    if not nodes:
        return ""
    if isinstance(nodes, dict):
        nodes = [nodes]
    if not isinstance(nodes, list):
        return str(nodes)

    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("@lang") == "en":
            target = node.get(nested_key, node) if nested_key else node
            if isinstance(target, dict):
                return target.get(value_key, "")
            return str(target)

    first = nodes[0] if nodes else {}

    if not isinstance(first, dict):
        return str(first)

    target = first.get(nested_key, first) if nested_key else first
    if isinstance(target, dict):
        return target.get(value_key, "")

    return str(target)


if __name__ == "__main__":
    token = get_epo_token()

    ids = search_and_get_ids(token, "ta=drone", count=2)
    print(f"found ids: {ids}")

    results = hydrate_patents(token, ids)
    print(json.dumps(results, indent=2))
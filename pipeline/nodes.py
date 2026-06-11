import re
import json
import time
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from epo_pipeline import (
    get_epo_token,
    search_and_get_ids,
    hydrate_patents,
    fetch_patent_claims,
)
from vectorengine import (
    initialize_vector_db,
    load_claims_to_db,
    retrieve_top_claims_per_patent,
)

from .state import FTOState
from .llm import llm, structured_risk_llm, structured_report_llm, call_llm_with_retry
from .routing import (
    sanitize_epo_query,
    _SIGNIFICANT_OVERLAP_THRESHOLD,
    merge_top_patent_ids,
    build_relevance_context,
)


def _llm_text(response) -> str:
    text = response.content if hasattr(response, "content") else str(response)
    if isinstance(text, list):
        text = " ".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in text
        )
    return text.strip()


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    return re.sub(r"\s*```$", "", raw)


def node_expand_queries(state: FTOState):
    retry_count = state.get("retry_count", 0)
    previous_queries = state.get("sub_queries") or []

    if retry_count > 0 and previous_queries:
        print(f"retry {retry_count}, generating new search queries...")
    else:
        print("generating search queries...")

    idea = state["user_idea"]

    retry_block = ""
    if retry_count > 0 and previous_queries:
        failed_list = "\n".join(f"  - {q}" for q in previous_queries)
        retry_block = (
            f"\n\n RETRY ATTEMPT {retry_count} — PREVIOUS SEARCH UNDERPERFORMED:\n"
            f"The following search phrases were already tried but did NOT return enough "
            f"relevant patents from EPO. Do NOT repeat them verbatim or with only minor "
            f"wording changes (e.g. swapping a single synonym or reordering words).\n"
            f"Previously tried (DO NOT reuse):\n{failed_list}\n\n"
            f"Generate 6 COMPLETELY NEW phrases using different engineering angles, "
            f"broader or alternative terminology, different mechanism descriptions, or "
            f"adjacent IPC conceptual spaces. Each new phrase must be substantively "
            f"different from every item above.\n"
        )

    prompt = (
        f"You are a patent claim drafter and prior-art search expert specialising in "
        f"US and European patent databases (USPTO, EPO).\n\n"
        f"Invention idea: \"{idea}\"\n\n"
        "Follow these two steps strictly:\n\n"
        "STEP 1 — Technical decomposition (internal reasoning only, do NOT output this):\n"
        "Break the invention down into its fundamental mechanical and technical concepts. "
        "Ask yourself: what physical principles, structural members, force-transfer mechanisms, "
        "material properties, or functional relationships make this invention work? "
        "Strip away all consumer product names, brand names, sport names, activity names, "
        "and everyday descriptors. Translate every concept into engineering or mechanical "
        "terminology that could appear verbatim in a patent claim "
        "(e.g. 'resilient member', 'elastic energy storage', 'vibration attenuation', "
        "'impact-absorbing substrate', 'compliant coupling element').\n\n"
        "DATABASE TERMINOLOGY RULES — critical for US/EP retrieval:\n"
        "  • Generate terms that would appear in US and European patent titles, "
        "NOT in consumer product descriptions or marketing copy.\n"
        "  • USPTO and EPO patents use formal engineering terminology. "
        "Think: mechanism descriptions, functional terms, material processes, "
        "engineering classifications — NOT brand names, sport names, or country-specific terms.\n"
        "  • Ask yourself: 'Would a USPTO examiner or EPO search examiner use this phrase "
        "as a title keyword?' If not, rephrase it.\n\n"
        "STEP 2 — Output exactly 6 search phrases structured as follows:\n"
        "  Query 1-2: Core mechanism terms (what the invention physically does, "
        "as a USPTO/EPO title would describe it)\n"
        "  Query 3: The broader technical category the invention belongs to "
        "(an engineering classification, not a product category)\n"
        "  Query 4: The problem the invention solves expressed as an engineering challenge "
        "(not the solution, not a product name)\n"
        "  Query 5: A related adjacent technology that might have overlapping claims\n"
        "  Query 6: The IPC (International Patent Classification) conceptual space this "
        "invention belongs to — express it as the functional engineering concept that defines "
        "the IPC subclass (e.g. 'unmanned aerial vehicle fluid dispensing apparatus', "
        "'elastic energy storage compliant joint mechanism'). This phrase should map to an "
        "IPC subgroup and use the formal language found in IPC class definitions.\n\n"
        "Example for 'drone that sprays paint':\n"
        "  1. aerial spray coating system\n"
        "  2. UAV fluid applicator nozzle\n"
        "  3. autonomous surface treatment apparatus\n"
        "  4. building facade coating automation\n"
        "  5. unmanned vehicle mounted fluid dispensing\n"
        "  6. unmanned aerial vehicle coating apparatus IPC B64C\n\n"
        "Hard rules for every phrase:\n"
        "  • Use engineering and mechanical terminology only — NO brand names, sport names, "
        "activity names, or consumer product terms.\n"
        "  • 2–5 words per phrase.\n"
        "  • No boolean operators, no special characters, no brackets, no quotes, no wildcards.\n\n"
        "Return ONLY a numbered list (1. … 2. … 3. … 4. … 5. … 6. …), "
        "one phrase per line, nothing else."
        f"{retry_block}"
    )

    text = _llm_text(call_llm_with_retry(llm, prompt))

    sub_queries: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.\)\:\-]\s*(.+)$", line)
        if match:
            sub_queries.append(match.group(1).strip())
        elif len(sub_queries) < 6:
            sub_queries.append(line)

    return {"sub_queries": sub_queries[:6]}


def node_retrieve_patents(state: FTOState):
    print("searching EPO...")
    token = get_epo_token()

    def _search(query: str) -> List[str]:
        safe = sanitize_epo_query(query)
        return search_and_get_ids(token, safe, count=8)

    per_query_ids: List[List[str]] = [[] for _ in state["sub_queries"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_search, q): i for i, q in enumerate(state["sub_queries"])
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                per_query_ids[i] = fut.result() or []
            except Exception as exc:
                print(f"query {i + 1} failed: {exc}")
                per_query_ids[i] = []

    top_20_ids = merge_top_patent_ids(per_query_ids, n=20)
    if not top_20_ids:
        print("no patents found")
        return {"raw_patents": []}

    print(f"selected {len(top_20_ids)} patent ids")

    hydrated = hydrate_patents(token, top_20_ids)
    results: List[dict] = [{**patent, "source": "epo"} for patent in hydrated]

    # EPO rate limit (~2.5 rps)
    dispatch_lock = threading.Lock()
    last_dispatch = [0.0]
    min_gap = 0.4

    def _fetch_claims(patent: dict) -> dict:
        pid = patent["id"]
        with dispatch_lock:
            now = time.time()
            gap = min_gap - (now - last_dispatch[0])
            if gap > 0:
                time.sleep(gap)
            last_dispatch[0] = time.time()

        claims = fetch_patent_claims(token, pid)
        out = dict(patent)
        if claims:
            out["claims_text"] = claims
            out["context_type"] = "claim"
            print(f"{pid}: got claims")
        else:
            out["claims_text"] = patent.get("text_for_vector_db", "")
            out["context_type"] = "abstract_fallback"
            print(f"{pid}: no claims, using abstract")
        return out

    enriched: List[dict] = [{}] * len(results)
    with ThreadPoolExecutor(max_workers=3) as pool:
        idx_futures = {pool.submit(_fetch_claims, p): i for i, p in enumerate(results)}
        for fut in as_completed(idx_futures):
            i = idx_futures[fut]
            try:
                enriched[i] = fut.result()
            except Exception as exc:
                print(f"claims fetch failed for patent {i}: {exc}")
                enriched[i] = results[i]

    final = [p for p in enriched if p]
    print(f"got {len(final)} patents total")
    return {"raw_patents": final}


def node_filter_relevant_patents(state: FTOState):
    print("scoring patent relevance...")
    user_idea = state["user_idea"]

    candidates = [
        p for p in state["raw_patents"]
        if (p.get("claims_text") or p.get("text_for_vector_db", "")).strip()
    ]

    if not candidates:
        print("nothing to score")
        return {"raw_patents": []}

    patent_lines = []
    for i, patent in enumerate(candidates, start=1):
        context_block = build_relevance_context(patent)
        patent_lines.append(f"{i}. {patent['id']}:\n{context_block}")

    prompt = (
        f"Score each patent's relevance to this invention: '{user_idea}'\n"
        "Focus on technical mechanism overlap, ignore domain/industry.\n\n"
        "Scoring rules:\n"
        "  • Ignore all product names, sport names, domain names, and brand names entirely.\n"
        "  • Focus ONLY on the underlying mechanical and functional concepts: "
        "structural members, force-transfer mechanisms, material behaviour, "
        "energy storage/dissipation, motion control, actuation principles, etc.\n"
        "  • Score 0 immediately if the patent is about software, biological processes, "
        "chemical compounds, electrical circuits with no mechanical elements, or pure data processing.\n"
        "  0–3  = completely different technical domain, no concept overlap\n"
        "  4–6  = shares some mechanical/functional concepts, worth analysing\n"
        "  7–10 = directly relevant technical overlap\n\n"
        f"Patents to score:\n" + "\n".join(patent_lines) + "\n\n"
        'Return ONLY a JSON array: [{"patent_id": "...", "score": 7}, ...]\n'
        "Scores 0-10. No explanation."
    )

    raw = _strip_json_fences(_llm_text(call_llm_with_retry(llm, prompt)))

    try:
        scored = json.loads(raw)
        score_map = {entry["patent_id"]: int(entry["score"]) for entry in scored}
    except Exception:
        print("json parse failed, defaulting all to score 5")
        for patent in candidates:
            patent["relevance_score"] = 5
        return {"raw_patents": candidates}

    relevant = []
    for patent in candidates:
        score = max(0, min(10, score_map.get(patent["id"], 0)))
        if score >= 5:
            patent["relevance_score"] = score
            relevant.append(patent)
            print(f"kept {patent['id']} (score {score})")
        else:
            print(f"dropped {patent['id']} (score {score})")

    relevant.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)
    relevant = relevant[:6]
    print(f"{len(relevant)} patents left after filter")
    return {"raw_patents": relevant}


def node_score_retrieval_quality(state: FTOState):
    count = len(state["raw_patents"])
    if count == 0:
        score = 0.0
    elif count <= 2:
        score = 0.4
    else:
        score = 0.8
    print(f"retrieval quality: {score} ({count} patents)")
    return {"retrieval_quality_score": score}


def node_increment_retry(state: FTOState):
    return {"retry_count": state["retry_count"] + 1}


def node_no_results(state: FTOState):
    print("no relevant patents found after retries")
    return {
        "final_report": (
            "No relevant patents found for this invention idea after exhaustive search. "
            "The invention may operate freely or the search terms need refinement."
        ),
        "risk_assessments": [],
        "cleared_patents": [],
    }


def node_seed_claims_to_chromadb(state: FTOState):
    print("loading claims into chromadb...")
    collection = initialize_vector_db(reset=True)
    load_claims_to_db(collection, state["raw_patents"])
    return {}


def node_retrieve_top_claims(state: FTOState):
    print("fetching top claims from chromadb...")
    collection = initialize_vector_db()
    patent_ids = [p["id"] for p in state["raw_patents"]]
    hits = retrieve_top_claims_per_patent(
        collection, state["user_idea"], patent_ids, n_per_patent=2
    )

    if not hits:
        print("chromadb empty, using abstracts")
        hits = [
            {
                "patent_id": p["id"],
                "patent_title": p.get("title", ""),
                "claim_number": 0,
                "claim_text": p.get("text_for_vector_db", ""),
            }
            for p in state["raw_patents"]
        ]

    context_type_map = {p["id"]: p.get("context_type", "claim") for p in state["raw_patents"]}

    grouped: dict = {}
    for hit in hits:
        pid = hit["patent_id"]
        hit_context_type = hit.get("context_type") or context_type_map.get(pid, "claim")
        if pid not in grouped:
            grouped[pid] = {
                "patent_id": pid,
                "patent_title": hit["patent_title"],
                "claims": [],
                "context_type": hit_context_type,
            }
        cn = hit["claim_number"]
        if cn == 0:
            label = "Abstract"
        elif hit_context_type == "unformatted_claims":
            label = "Claims (raw)"
        else:
            label = f"Claim {cn}"
        grouped[pid]["claims"].append(f"{label}: {hit['claim_text']}")

    grouped_list = list(grouped.values())
    print(f"{len(grouped_list)} patents ready for risk check")
    return {"decomposed_claims": grouped_list}


def node_assess_risk(state: FTOState):
    print("checking infringement risk...")
    results: List[dict] = []

    max_single_claim = 5_000
    max_claim_block = 10_000

    for item in state["decomposed_claims"]:
        safe_claims = []
        for claim_str in item["claims"]:
            if len(claim_str) > max_single_claim:
                claim_str = claim_str[:max_single_claim] + " [...truncated...]"
                print(f"{item['patent_id']}: claim too long, truncated")
            safe_claims.append(claim_str)

        joined = "\n".join(safe_claims)
        if len(joined) > max_claim_block:
            claims_block = joined[:max_claim_block] + "\n[...truncated...]"
            print(f"{item['patent_id']}: claims truncated")
        else:
            claims_block = joined

        try:
            context_type = item.get("context_type", "claim")

            if context_type == "abstract_fallback":
                prompt = (
                    "You are a senior patent attorney performing a Freedom to Operate assessment. "
                    "Respond ONLY with a valid JSON object — no explanation, no markdown fences.\n\n"
                    "⚠️  ABSTRACT-ONLY MODE — STRICT RULES APPLY:\n"
                    "The text below is a patent ABSTRACT SUMMARY. Full legal claim text was NOT "
                    "retrieved. You have NO access to the actual patent claims.\n\n"
                    "ABSOLUTE PROHIBITIONS (any violation is an error):\n"
                    "  • Do NOT reference claim numbers (e.g. 'Claim 1').\n"
                    "  • Do NOT evaluate specific claim limitations.\n"
                    "  • Do NOT flag overlap based on shared keywords or generic phrases.\n"
                    "  • Do NOT use vague phrases such as 'shares similarities' or 'may overlap'.\n\n"
                    "ELEMENT MAPPING — REQUIRED METHODOLOGY (apply even to abstract text):\n"
                    "  1. Mentally decompose the user's invention into its core structural components.\n"
                    "  2. Identify the structural/functional scope described in the abstract.\n"
                    "  3. Determine whether the abstract describes physical mechanics that would "
                    "replicate or be equivalent to the user invention's components.\n\n"
                    "REASONING FORMAT — your reasoning field MUST:\n"
                    "  • Begin with this exact disclaimer (copy verbatim): '[DISCLAIMER: Full claim "
                    "text unavailable — assessment based on abstract summary only. A definitive FTO "
                    "determination requires retrieval of the full patent claims.]'\n"
                    "  • Follow with one sentence identifying the core structural technology described "
                    "in the abstract and stating concretely whether it maps onto the user invention's "
                    "components — naming those components explicitly.\n\n"
                    f"User invention: '{state['user_idea']}'\n"
                    f"Patent abstract:\n{claims_block}\n\n"
                    "Return JSON with exactly these keys: risk_level (HIGH/MEDIUM/LOW), "
                    "overlap_score (float 0.0 to 1.0), reasoning (string)."
                )

            elif context_type == "unformatted_claims":
                prompt = (
                    "You are a senior patent attorney performing a Freedom to Operate assessment. "
                    "Respond ONLY with a valid JSON object — no explanation, no markdown fences.\n\n"
                    "⚠️  UNFORMATTED CLAIMS MODE — STRICT RULES APPLY:\n"
                    "The text below is REAL PATENT CLAIM LANGUAGE from the patent database in a "
                    "non-standard layout. Individual claim numbers are NOT reliably delineated.\n\n"
                    "ABSOLUTE PROHIBITIONS (any violation is an error):\n"
                    "  • Do NOT cite or fabricate specific claim numbers.\n"
                    "  • Do NOT imply claims are absent — they ARE present below.\n"
                    "  • Do NOT base your score on shared generic keywords or isolated phrases.\n"
                    "  • Do NOT use vague phrases such as 'shares similarities' or 'may overlap'.\n\n"
                    "ELEMENT MAPPING — REQUIRED METHODOLOGY:\n"
                    "  1. Decompose the user's invention into its core structural components.\n"
                    "  2. Scan the claim text for structural/mechanical elements — not keywords.\n"
                    "  3. Determine whether claimed physical mechanics are equivalent to any of the "
                    "user invention's structural components.\n\n"
                    "REASONING FORMAT — your reasoning field MUST:\n"
                    "  • Begin with this exact disclaimer (copy verbatim): '[DISCLAIMER: Claims "
                    "retrieved in unformatted layout — individual claim numbers not cited. Assessment "
                    "is based on the full claim text provided.]'\n"
                    "  • Follow with one sentence that names a specific structural mechanism found in "
                    "the claim text and states explicitly whether it is mechanically equivalent to, or "
                    "distinct from, the user invention's components — giving a concrete physical reason.\n\n"
                    f"User invention: '{state['user_idea']}'\n"
                    f"Patent claim text:\n{claims_block}\n\n"
                    "Return JSON with exactly these keys: risk_level (HIGH/MEDIUM/LOW), "
                    "overlap_score (float 0.0 to 1.0), reasoning (string)."
                )

            else:
                prompt = (
                    "You are a senior patent attorney performing a Freedom to Operate assessment. "
                    "Respond ONLY with a valid JSON object — no explanation, no markdown fences.\n\n"
                    "The following are authoritative PATENT CLAIMS retrieved from the EPO database.\n\n"
                    "ELEMENT MAPPING FRAMEWORK — THIS IS YOUR REQUIRED METHODOLOGY:\n"
                    "  1. Decompose the user's invention into its core structural components "
                    "(e.g., for 'a drone that spray-paints': Component 1 = multi-rotor aerial chassis, "
                    "Component 2 = pressurized paint delivery nozzle assembly).\n"
                    "  2. Map each component against the physical/mechanical elements stated in the claims.\n"
                    "  3. Assess whether claimed elements are structurally equivalent to, or mechanically "
                    "distinct from, the user invention's components.\n\n"
                    "ABSOLUTE PROHIBITIONS (any violation is an error):\n"
                    "  • Do NOT base your score on shared generic keywords or isolated phrases "
                    "(e.g., do not flag a patent purely because it mentions 'fluid delivery' or 'spring').\n"
                    "  • Do NOT use vague phrases such as 'shares similarities', 'may overlap', "
                    "'contains relevant elements', or 'similar technology'.\n"
                    "  • Do NOT default to Claim 1 unless it is genuinely the most relevant claim.\n\n"
                    "DOMAIN RULE: Infringement is determined by mechanical element equivalence, NOT by "
                    "application domain or industry. A spring mechanism in construction equipment CAN "
                    "infringe a cricket bat spring if the physical mechanics are identical.\n\n"
                    f"User invention: '{state['user_idea']}'\n"
                    f"Patent claims:\n{claims_block}\n\n"
                    "Return JSON with exactly these keys: risk_level (HIGH/MEDIUM/LOW), "
                    "overlap_score (float 0.0 to 1.0), reasoning (2–3 sentences).\n\n"
                    "REASONING FORMAT — your reasoning field MUST:\n"
                    "  • Cite the specific claim number being evaluated (e.g., 'Claim 3').\n"
                    "  • Name which structural component of the user's invention maps to which specific "
                    "mechanical element in that claim.\n"
                    "  • State one concrete physical distinction or equivalence — e.g., 'the patent claims "
                    "an agricultural boom arm with downward spray dispersion; the user invention uses a "
                    "lateral nozzle aimed at vertical surfaces — mechanically distinct and non-infringing.'"
                )

            result = call_llm_with_retry(structured_risk_llm, prompt)
            assessment = {
                "patent_id": item["patent_id"],
                "risk_level": result.risk_level,
                "overlap_score": result.overlap_score,
                "reasoning": result.reasoning,
            }
        except Exception as exc:
            print(f"risk check failed for {item['patent_id']}: {exc}")
            assessment = {
                "patent_id": item["patent_id"],
                "risk_level": "UNKNOWN",
                "overlap_score": 0.0,
                "reasoning": f"Assessment unavailable: {type(exc).__name__}",
            }
        results.append(assessment)

    return {"risk_assessments": results}


def node_human_review(state: FTOState):
    return {"human_approved": True}


def node_write_report(state: FTOState):
    print("writing final report...")

    all_assessments = state["risk_assessments"]
    bucket_a = [
        a for a in all_assessments
        if (a.get("overlap_score") or 0) > _SIGNIFICANT_OVERLAP_THRESHOLD
    ]
    bucket_b = [
        a for a in all_assessments
        if (a.get("overlap_score") or 0) <= _SIGNIFICANT_OVERLAP_THRESHOLD
    ]

    print(f"high overlap: {len(bucket_a)}, cleared: {len(bucket_b)}")

    if not bucket_a:
        no_risk_prompt = (
            "You are a senior patent attorney drafting a formal Freedom to Operate clearance "
            "memorandum. Your response MUST be a valid JSON object matching the required schema "
            "— no markdown fences, no extra keys.\n\n"
            "=== SCHEMA ===\n"
            '{"report": "<markdown string>", "assessments": []}\n\n'
            "=== REPORT INSTRUCTIONS ===\n"
            f"Invention: {state['user_idea']}\n\n"
            "No patents with significant structural overlap (score > 0.2) were identified after "
            "exhaustive prior-art analysis. Write a concise, professionally worded FTO clearance "
            "statement in markdown.\n\n"
            "CRITICAL WRITING DIRECTIVE — Professional Legal Synthesis:\n"
            "  • Write in a definitive, legally rigorous tone — no hedging language.\n"
            "  • Do NOT invent or reference any patent IDs.\n"
            "  • The clearance statement should briefly characterise the user's invention by its "
            "core structural components and state that no claims with mechanically equivalent "
            "elements were identified in the assessed patent corpus.\n\n"
            "Keep the `assessments` array empty ([]).\n"
            "Structure: Executive Summary, Recommended Next Steps."
        )
        result = call_llm_with_retry(structured_report_llm, no_risk_prompt)
        return {
            "final_report": result.report,
            "risk_assessments": [],
            "cleared_patents": bucket_b,
        }

    structured_summary = "\n".join(
        f"Patent {a['patent_id']}: {a['risk_level']} risk "
        f"(overlap: {a['overlap_score']}) — {a['reasoning']}"
        for a in bucket_a
    )
    assessments_json = json.dumps(bucket_a, indent=2)
    patent_count = len(bucket_a)

    prompt = (
        "You are a senior patent attorney drafting a formal Executive Freedom to Operate "
        "Analysis Report. Your response MUST be a valid JSON object matching the required "
        "schema — no markdown fences, no extra keys.\n\n"
        "=== SCHEMA ===\n"
        '{"report": "<markdown string>", "assessments": [<array — see below>]}\n\n'
        "=== REPORT INSTRUCTIONS ===\n"
        "Write a professional FTO analysis report in markdown. You are analyzing ONLY patents "
        "with significant structural overlap (score > 0.2). Do NOT reference or invent any "
        "patents not listed below.\n\n"
        "CRITICAL — COMPLETENESS RULE:\n"
        f"You have been provided with exactly {patent_count} significant patent(s). "
        "You MUST discuss every single one in your report.\n\n"
        "CRITICAL WRITING DIRECTIVE — Professional Legal Synthesis:\n"
        "  • You are STRICTLY FORBIDDEN from using shallow descriptions such as 'due to the "
        "phrase X' or 'because it contains the keyword Y'.\n"
        "  • For every patent discussed, you MUST synthesise the structural overlap: explain "
        "the technical interaction between the patent's claimed mechanical elements and the "
        "user invention's core structural components.\n"
        "  • Example of FORBIDDEN writing: 'Patent X presents risk due to the phrase "
        "\"aerial fluid delivery\".'\n"
        "  • Example of REQUIRED writing: 'Patent X presents a HIGH risk because it claims "
        "an unmanned aerial platform structurally integrated with a surface-targeted liquid "
        "projection assembly — directly replicating the foundational multi-component mechanics "
        "of the user's invention.'\n"
        "  • Maintain a highly professional, definitive, and legally rigorous tone throughout. "
        "Avoid hedging language (e.g., 'may', 'could', 'might potentially').\n"
        "  • Each patent's risk must be characterised by mechanical element analysis, not "
        "keyword proximity.\n\n"
        f"Invention: {state['user_idea']}\n"
        f"Significant Patent Risk Assessments ({patent_count} total):\n"
        f"{structured_summary}\n\n"
        "Structure the report with: Executive Summary, Key Risk Patents, "
        "and Recommended Next Steps.\n\n"
        "=== ASSESSMENTS INSTRUCTIONS ===\n"
        "For the `assessments` output field copy the following JSON array EXACTLY "
        "as-is — do NOT modify, summarise, reorder, or omit any entry:\n"
        f"{assessments_json}"
    )

    result = call_llm_with_retry(structured_report_llm, prompt)

    llm_assessments = [a.model_dump() for a in result.assessments]
    if len(llm_assessments) < patent_count:
        print(
            f"report missing {patent_count - len(llm_assessments)} assessments, filling in"
        )
        llm_ids = {a["patent_id"] for a in llm_assessments}
        for original in bucket_a:
            if original["patent_id"] not in llm_ids:
                llm_assessments.append(original)

    return {
        "final_report": result.report,
        "risk_assessments": llm_assessments,
        "cleared_patents": bucket_b,
    }

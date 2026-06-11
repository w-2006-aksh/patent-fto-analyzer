from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import app, memory

api = FastAPI(title="Patent FTO API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    idea: str


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return (
        "429" in s
        or "rate_limit" in s.lower()
        or "RateLimitError" in type(exc).__name__
        or "RESOURCE_EXHAUSTED" in s
    )


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/start_analysis")
def start_analysis(request: AnalysisRequest):
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_idea": request.idea,
        "sub_queries": [],
        "raw_patents": [],
        "decomposed_claims": [],
        "risk_assessments": [],
        "cleared_patents": [],
        "retrieval_quality_score": 0.0,
        "retry_count": 0,
        "human_approved": False,
        "final_report": "",
    }

    try:
        phase1_state = app.invoke(initial_state, config=config)
    except Exception as e:
        if _is_rate_limit(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM rate limit reached during patent analysis. "
                    "Please wait a moment and try again."
                ),
            )
        raise HTTPException(status_code=500, detail=str(e))

    # no results. skip review, report is already done
    if phase1_state.get("final_report"):
        return JSONResponse(content={
            "thread_id": thread_id,
            "idea": request.idea,
            "complete": True,
            "risk_assessments": phase1_state.get("risk_assessments", []),
            "cleared_patents": phase1_state.get("cleared_patents", []),
            "report": phase1_state["final_report"],
        })

    return JSONResponse(content={
        "thread_id": thread_id,
        "idea": request.idea,
        "complete": False,
        "risk_assessments": phase1_state.get("risk_assessments", []),
    })


@api.post("/approve_analysis")
def approve_analysis(request: ApproveRequest):
    if not request.approved:
        return JSONResponse(content={
            "cancelled": True,
            "message": "Analysis cancelled.",
        })

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_state = app.invoke(None, config=config)
    except Exception as e:
        if _is_rate_limit(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM rate limit reached while writing the final report. "
                    "Please wait a moment and try again."
                ),
            )
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "idea": final_state.get("user_idea", ""),
        "risk_assessments": final_state.get("risk_assessments", []),
        "cleared_patents": final_state.get("cleared_patents", []),
        "report": final_state.get("final_report", ""),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)

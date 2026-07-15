import re
import threading
import time
from collections import deque
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .state import RiskAssessment, FTOReport, RelevanceBatch

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_risk_llm = llm.with_structured_output(RiskAssessment, method="json_mode")
structured_report_llm = llm.with_structured_output(FTOReport, method="json_mode")
structured_relevance_llm = llm.with_structured_output(RelevanceBatch, method="json_mode")

_LLM_CALL_DELAY = 3
_last_llm_call_time: float = 0.0

# groq tpm cap
_TPM_LIMIT = 10_000
_TPM_WINDOW_SEC = 60.0
_tpm_history: deque[tuple[float, int]] = deque()
_CHARS_PER_TOKEN = 4
_llm_lock = threading.Lock()


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc)
    return (
        "429" in s
        or "rate_limit" in s.lower()
        or "RateLimitError" in type(exc).__name__
        or "RESOURCE_EXHAUSTED" in s
    )


def _is_structured_output_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in ("ValidationError", "OutputParserException", "JSONDecodeError", "ParserError"):
        return True
    s = str(exc).lower()
    return "validation error" in s or "failed to parse" in s or "json" in s


def _throttle_llm_call():
    global _last_llm_call_time
    elapsed = time.time() - _last_llm_call_time
    if elapsed < _LLM_CALL_DELAY:
        time.sleep(_LLM_CALL_DELAY - elapsed)
    _last_llm_call_time = time.time()


def _prune_tpm_history() -> int:
    now = time.time()
    while _tpm_history and now - _tpm_history[0][0] >= _TPM_WINDOW_SEC:
        _tpm_history.popleft()
    return sum(tokens for _, tokens in _tpm_history)


def _check_tpm_budget(prompt: str, response_budget: int = 400):
    estimated = len(prompt) // _CHARS_PER_TOKEN + response_budget
    used = _prune_tpm_history()

    while used + estimated > _TPM_LIMIT and _tpm_history:
        oldest_ts = _tpm_history[0][0]
        wait = _TPM_WINDOW_SEC - (time.time() - oldest_ts) + 2
        if wait > 0:
            print(f"tpm limit hit, waiting {wait:.0f}s...")
            time.sleep(wait)
        used = _prune_tpm_history()

    _tpm_history.append((time.time(), estimated))


def call_llm_with_retry(llm_instance, prompt, max_retries: int = 5):
    global _last_llm_call_time

    with _llm_lock:
        prompt_str = str(prompt)
        _throttle_llm_call()
        _check_tpm_budget(prompt_str)

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                result = llm_instance.invoke(prompt)
                _last_llm_call_time = time.time()
                return result
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    delay_match = re.search(
                        r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE
                    )
                    suggested = float(delay_match.group(1)) if delay_match else 0
                    backoff = max(suggested + 5, 62)
                    print(f"rate limit, waiting {backoff:.0f}s (attempt {attempt})...")
                    time.sleep(backoff)
                    _last_llm_call_time = time.time()
                    _tpm_history.clear()
                    last_exc = exc
                else:
                    raise
        raise last_exc


def call_structured_with_retry(llm_instance, prompt, max_retries: int = 3):
    """Structured LLM call with retries on rate limits and parse/validation failures."""
    global _last_llm_call_time

    with _llm_lock:
        prompt_str = str(prompt)
        last_exc = None

        for attempt in range(1, max_retries + 1):
            _throttle_llm_call()
            _check_tpm_budget(prompt_str, response_budget=800)

            try:
                result = llm_instance.invoke(prompt)
                _last_llm_call_time = time.time()
                return result
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc):
                    delay_match = re.search(
                        r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE
                    )
                    suggested = float(delay_match.group(1)) if delay_match else 0
                    backoff = max(suggested + 5, 62)
                    print(f"rate limit, waiting {backoff:.0f}s (attempt {attempt})...")
                    time.sleep(backoff)
                    _last_llm_call_time = time.time()
                    _tpm_history.clear()
                elif _is_structured_output_error(exc) and attempt < max_retries:
                    print(f"structured output failed ({type(exc).__name__}), retry {attempt}...")
                    time.sleep(2)
                else:
                    raise

        raise last_exc

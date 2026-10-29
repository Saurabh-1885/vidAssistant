# Actionable items , decisions , questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    # Cached shared client; limit SDK retries so rate-limit errors fail fast
    # instead of silently re-sending the request multiple times.
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature= 0.2, max_retries=1)


def build_chain(system_prompt : str):
    llm  = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),

    ]) | llm | StrOutputParser()
    )


# ─── Combined Extraction ──────────────────────────────────────────────────────
# Previously each of the three extractions (action items, decisions, questions)
# sent the FULL transcript to Mistral in its own API call (3 calls per run).
# They are now merged into ONE call that returns all three sections separated
# by markers, which we split back out locally — cutting extraction API calls
# from 3 to 1 with no loss of information.

COMBINED_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. From the meeting transcript, extract "
    "three things and return them in EXACTLY this format:\n\n"
    "=== ACTION ITEMS ===\n"
    "<numbered list; for each item provide:\n"
    "- Task description\n"
    "- Owner (who is responsible)\n"
    "- Deadline (if mentioned, else write 'Not specified')\n"
    "If none found write exactly 'No action items found.'\n\n"
    "=== KEY DECISIONS ===\n"
    "<numbered list of all key decisions made>\n"
    "If none found write exactly 'No key decisions found.'\n\n"
    "=== OPEN QUESTIONS ===\n"
    "<numbered list of all unsolved questions or topics needing follow-up>\n"
    "If none found write exactly 'No open questions found.'\n\n"
    "Rules:\n"
    "- Use ONLY information from the transcript. Do not invent information.\n"
    "- Output ONLY the three marked sections above, nothing else."
)

_SECTION_MARKERS = ["ACTION ITEMS", "KEY DECISIONS", "OPEN QUESTIONS"]


def extract_all(transcript: str) -> dict:
    """Single Mistral call that extracts action items, decisions and questions.

    Returns a dict with keys: action_items, key_decisions, open_questions.
    """
    raw = build_chain(COMBINED_SYSTEM_PROMPT).invoke(transcript)

    # Split the response on the section markers.
    parts = {}
    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        matched = next((m for m in _SECTION_MARKERS if f"=== {m} ===" in stripped), None)
        if matched:
            current = matched
            parts[current] = []
        elif current is not None:
            parts[current].append(line)

    def clean(marker: str, fallback: str) -> str:
        text = "\n".join(parts.get(marker, [])).strip()
        return text if text else fallback

    return {
        "action_items": clean("ACTION ITEMS", "No action items found."),
        "key_decisions": clean("KEY DECISIONS", "No key decisions found."),
        "open_questions": clean("OPEN QUESTIONS", "No open questions found."),
    }


# ─── Backwards-compatible wrappers ────────────────────────────────────────────
# These delegate to the combined extraction so existing callers keep working.
# NOTE: calling more than one of these individually re-runs the combined call;
# prefer extract_all() directly when you need all three.

_last_extraction_cache = {}


def _get_extraction(transcript: str) -> dict:
    # Simple memoization keyed on the transcript so the three wrapper
    # functions below share ONE API call if used together in a run.
    key = hash(transcript)
    if key not in _last_extraction_cache:
        _last_extraction_cache.clear()  # keep only the latest transcript
        _last_extraction_cache[key] = extract_all(transcript)
    return _last_extraction_cache[key]


def extract_action_items(transcript: str) -> str:
    return _get_extraction(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return _get_extraction(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return _get_extraction(transcript)["open_questions"]
    
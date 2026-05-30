import re


BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"reveal system prompt",
    r"show system prompt",
    r"bypass security",
    r"jailbreak",
    r"developer mode",
    r"act as unrestricted",
    r"disable safety",
]


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate user query before retrieval and LLM generation.
    """

    if not query or not query.strip():
        return False, "Query cannot be empty."

    cleaned_query = query.lower().strip()

    if len(cleaned_query) < 3:
        return False, "Query is too short."

    if len(cleaned_query) > 1000:
        return False, "Query is too long."

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cleaned_query):
            return False, "Potential prompt injection detected."

    return True, "Query passed guardrails."


def validate_response(response: str) -> str:
    """
    Validate generated LLM response before sending it to the user.
    """

    if not response or not response.strip():
        return "I could not generate a response."

    blocked_terms = [
        "system prompt",
        "ignore previous instructions",
        "developer mode"
    ]

    cleaned_response = response.strip()

    for term in blocked_terms:
        if term in cleaned_response.lower():
            return "Response blocked due to safety policy."

    return cleaned_response
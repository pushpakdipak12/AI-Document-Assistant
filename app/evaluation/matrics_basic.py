def keyword_match_score(text: str, keywords: list[str]) -> float:
    """
    Keyword-based score between 0 and 1.
    """

    if not text or not keywords:
        return 0.0

    text = text.lower()

    matches = 0

    for keyword in keywords:
        if keyword.lower() in text:
            matches += 1

    return round(matches / len(keywords), 3)


def retrieval_score(context: str, keywords: list[str]) -> float:
    """
    Measures whether retrieved context contains expected concepts.
    """

    return keyword_match_score(
        text=context,
        keywords=keywords
    )


def answer_score(answer: str, keywords: list[str]) -> float:
    """
    Measures whether final answer contains expected concepts.
    """

    return keyword_match_score(
        text=answer,
        keywords=keywords
    )


def latency_score(start: float, end: float) -> float:
    """
    Latency in seconds.
    """

    return round(end - start, 3)
from groq import Groq

from app.core.config import settings


def _get_client() -> Groq:
    """
    Create Groq client safely.
    """

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please add it inside backend/.env"
        )

    return Groq(
        api_key=settings.GROQ_API_KEY
    )


def _clean_context(context: str) -> str:
    """
    Clean retrieved context before sending it to the LLM.
    """

    context = context.replace("\n", " ")
    context = " ".join(context.split())

    return context


def _build_prompt(
    query: str,
    context: str
) -> str:
    """
    Build grounded RAG prompt.
    """

    return f"""
You are an expert AI assistant.

Your job is to answer ONLY from the uploaded document context.

Rules:
- Do not use outside knowledge.
- Do not hallucinate.
- Do not invent details.
- If the answer is not available in the context, say:
  "I could not find the answer in the uploaded documents."
- Keep the answer clear, practical, and professional.
- Use markdown formatting.
- Prefer concise explanation with useful bullet points.
- Mention only information supported by the context.

Context:
{context}

Question:
{query}

Generate a well-structured answer.
"""


def generate_answer(
    query: str,
    context: str,
    temperature: float = 0.3,
    max_tokens: int = 700
) -> str:
    """
    Generate grounded answer using Groq LLM.

    temperature:
        Lower value = more factual/stable.
        Higher value = more creative.

    max_tokens:
        Maximum output length.
    """

    client = _get_client()

    cleaned_context = _clean_context(
        context
    )

    prompt = _build_prompt(
        query=query,
        context=cleaned_context
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content
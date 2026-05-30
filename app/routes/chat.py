import math
from fastapi import APIRouter, HTTPException

from app.models.chat_schema import ChatRequest

from app.services.guardrail_service import (
    validate_query,
    validate_response
)

from app.services.hybrid_retrieval_service import hybrid_search
from app.services.reranker_service import rerank_chunks
from app.services.llm_service import generate_answer

from app.services.memory_service import (
    get_conversation_history,
    save_message
)


router = APIRouter(
    tags=["Chat"]
)


def _sigmoid(value: float) -> float:
    """
    Convert reranker score into 0-1 confidence-like value.
    """
    try:
        return 1 / (1 + math.exp(-value))
    except OverflowError:
        return 0.0


def _build_retrieval_metrics(chunks: list[dict]) -> dict:
    """
    Build lightweight retrieval quality indicators.
    """

    if not chunks:
        return {
            "retrieved_count": 0,
            "used_chunks": 0,
            "top_rerank_score": 0.0,
            "retrieval_confidence": 0.0,
            "avg_vector_score": 0.0,
            "avg_bm25_score": 0.0
        }

    rerank_scores = [
        float(item.get("rerank_score", 0.0))
        for item in chunks
    ]

    vector_scores = [
        float(item.get("vector_score", 0.0))
        for item in chunks
    ]

    bm25_scores = [
        float(item.get("bm25_score", 0.0))
        for item in chunks
    ]

    top_rerank_score = max(rerank_scores)

    retrieval_confidence = _sigmoid(top_rerank_score)

    return {
        "retrieved_count": len(chunks),
        "used_chunks": len(chunks),
        "top_rerank_score": round(top_rerank_score, 4),
        "retrieval_confidence": round(retrieval_confidence, 4),
        "avg_vector_score": round(sum(vector_scores) / len(vector_scores), 4),
        "avg_bm25_score": round(sum(bm25_scores) / len(bm25_scores), 4)
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    """
    Chat over uploaded documents with:
    - guardrails
    - hybrid retrieval
    - reranking
    - conversation memory
    - configurable temperature/max_tokens
    - retrieval metrics
    """

    try:
        session_id = (
            payload.session_id
            or "demo"
        ).strip()

        query = payload.query.strip()

        is_valid, guardrail_message = validate_query(query)

        if not is_valid:
            return {
                "session_id": session_id,
                "query": query,
                "answer": guardrail_message,
                "retrieved_chunks": [],
                "retrieval_type": "hybrid_bm25_vector",
                "bm25_used": False,
                "reranking_used": False,
                "memory_used": False,
                "guardrail_status": "blocked",
                "guardrail_message": guardrail_message,
                "llm_parameters": {
                    "temperature": payload.temperature,
                    "max_tokens": payload.max_tokens
                },
                "retrieval_metrics": {
                    "retrieved_count": 0,
                    "used_chunks": 0,
                    "top_rerank_score": 0.0,
                    "retrieval_confidence": 0.0,
                    "avg_vector_score": 0.0,
                    "avg_bm25_score": 0.0
                }
            }

        previous_messages = get_conversation_history(
            session_id
        )[-4:]

        history_text = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in previous_messages
            ]
        )

        retrieved_chunks = hybrid_search(
            query=query,
            top_k=payload.top_k
        )

        if not retrieved_chunks:
            answer = "No documents found. Please upload a PDF first."

            save_message(
                session_id,
                "user",
                query
            )

            save_message(
                session_id,
                "assistant",
                answer
            )

            return {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "retrieved_chunks": [],
                "retrieval_type": "hybrid_bm25_vector",
                "bm25_used": False,
                "reranking_used": False,
                "memory_used": len(previous_messages) > 0,
                "guardrail_status": "passed",
                "guardrail_message": guardrail_message,
                "llm_parameters": {
                    "temperature": payload.temperature,
                    "max_tokens": payload.max_tokens
                },
                "retrieval_metrics": _build_retrieval_metrics([])
            }

        reranked_chunks = rerank_chunks(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=payload.final_k
        )

        document_context = "\n\n".join(
            [
                item["chunk"]
                for item in reranked_chunks
            ]
        )

        full_context = f"""
Conversation History:
{history_text}

Document Context:
{document_context}
"""

        raw_answer = generate_answer(
            query=query,
            context=full_context,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )

        final_answer = validate_response(raw_answer)

        save_message(
            session_id,
            "user",
            query
        )

        save_message(
            session_id,
            "assistant",
            final_answer
        )

        retrieval_metrics = _build_retrieval_metrics(
            reranked_chunks
        )

        return {
            "session_id": session_id,
            "query": query,
            "answer": final_answer,
            "retrieved_chunks": reranked_chunks,
            "retrieval_type": "hybrid_bm25_vector",
            "bm25_used": any(
                item.get("bm25_score", 0) > 0
                for item in reranked_chunks
            ),
            "reranking_used": True,
            "memory_used": len(previous_messages) > 0,
            "guardrail_status": "passed",
            "guardrail_message": guardrail_message,
            "llm_parameters": {
                "temperature": payload.temperature,
                "max_tokens": payload.max_tokens
            },
            "retrieval_metrics": retrieval_metrics
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"chat_pipeline_failed: {error}"
        )
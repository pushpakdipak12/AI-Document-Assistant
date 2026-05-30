import time

from app.evaluation.dataset import EVAL_DATASET

from app.evaluation.matrics_basic import (
    retrieval_score,
    answer_score,
    latency_score
)

from app.services.hybrid_retrieval_service import hybrid_search
from app.services.reranker_service import rerank_chunks
from app.services.llm_service import generate_answer


def run_hybrid_evaluation(
    temperature: float = 0.3,
    max_tokens: int = 700,
    top_k: int = 10,
    final_k: int = 3
):
    """
    Run custom RAG evaluation without RAGAS.

    This evaluates:
    - retrieval score
    - answer score
    - latency
    - per-question errors safely
    """

    results = []

    total_retrieval = 0.0
    total_answer = 0.0
    total_latency = 0.0
    successful_questions = 0

    for item in EVAL_DATASET:

        query = item["question"]
        expected_keywords = item["expected_keywords"]

        start_time = time.time()

        try:
            retrieved_chunks = hybrid_search(
                query=query,
                top_k=top_k
            )

            reranked_chunks = rerank_chunks(
                query=query,
                retrieved_chunks=retrieved_chunks,
                top_k=final_k
            )

            context = "\n\n".join(
                [
                    chunk["chunk"]
                    for chunk in reranked_chunks
                ]
            )

            if context.strip():
                answer = generate_answer(
                    query=query,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                answer = "I could not find the answer in the uploaded documents."

            end_time = time.time()

            r_score = retrieval_score(
                context,
                expected_keywords
            )

            a_score = answer_score(
                answer,
                expected_keywords
            )

            l_score = latency_score(
                start_time,
                end_time
            )

            total_retrieval += r_score
            total_answer += a_score
            total_latency += l_score
            successful_questions += 1

            results.append(
                {
                    "query": query,
                    "expected_keywords": expected_keywords,
                    "retrieval_score": r_score,
                    "answer_score": a_score,
                    "latency_seconds": l_score,
                    "retrieved_chunks_count": len(reranked_chunks),
                    "status": "success",
                    "error": None
                }
            )

        except Exception as error:

            end_time = time.time()

            results.append(
                {
                    "query": query,
                    "expected_keywords": expected_keywords,
                    "retrieval_score": 0.0,
                    "answer_score": 0.0,
                    "latency_seconds": round(end_time - start_time, 3),
                    "retrieved_chunks_count": 0,
                    "status": "failed",
                    "error": str(error)
                }
            )

    total_questions = len(EVAL_DATASET)

    if successful_questions == 0:
        average_retrieval_score = 0.0
        average_answer_score = 0.0
        average_latency_seconds = 0.0
    else:
        average_retrieval_score = round(
            total_retrieval / successful_questions,
            3
        )

        average_answer_score = round(
            total_answer / successful_questions,
            3
        )

        average_latency_seconds = round(
            total_latency / successful_questions,
            3
        )

    return {
        "evaluation_type": "custom_no_ragas",
        "average_retrieval_score": average_retrieval_score,
        "average_answer_score": average_answer_score,
        "average_latency_seconds": average_latency_seconds,
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "failed_questions": total_questions - successful_questions,
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "final_k": final_k
        },
        "details": results
    }
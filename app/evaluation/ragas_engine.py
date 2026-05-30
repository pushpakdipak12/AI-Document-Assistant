from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.services.hybrid_retrieval_service import hybrid_search
from app.services.reranker_service import rerank_chunks
from app.services.llm_service import generate_answer


def build_dataset(test_cases: list) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_cases:
        query = item["question"]
        expected_keywords = item["expected_keywords"]

        retrieved = hybrid_search(query=query, top_k=10)
        reranked = rerank_chunks(
            query=query,
            retrieved_chunks=retrieved,
            top_k=3
        )

        context_texts = [c["chunk"] for c in reranked]

        if context_texts:
            answer = generate_answer(
                query=query,
                context="\n\n".join(context_texts)
            )
        else:
            answer = "I could not find the answer in the uploaded documents."

        questions.append(query)
        answers.append(answer)
        contexts.append(context_texts)

        # RAGAS expects ground truth-like text
        ground_truths.append([
            f"The answer should mention: {', '.join(expected_keywords)}."
        ])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    })


def run_ragas_eval(test_cases: list) -> dict:
    """
    Return RAGAS metrics in JSON-friendly form.
    """
    try:
        dataset = build_dataset(test_cases)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        try:
            # Convert to JSON-friendly output for FastAPI / Streamlit
            return {
                "rows": result.to_pandas().to_dict(orient="records")
            }
        except Exception:
            return {
                "raw": str(result)
            }

    except Exception as e:
        return {
            "error": f"RAGAS evaluation failed: {e}"
        }
from app.evaluation.evaluator import run_hybrid_evaluation


if __name__ == "__main__":

    report = run_hybrid_evaluation()

    print("\n===== CUSTOM RAG EVALUATION REPORT =====\n")

    print("Evaluation Type:", report["evaluation_type"])
    print("Average Retrieval Score:", report["average_retrieval_score"])
    print("Average Answer Score:", report["average_answer_score"])
    print("Average Latency:", report["average_latency_seconds"])
    print("Total Questions:", report["total_questions"])
    print("Successful Questions:", report["successful_questions"])
    print("Failed Questions:", report["failed_questions"])

    print("\n--- Detailed Results ---\n")

    for item in report["details"]:
        print(item)
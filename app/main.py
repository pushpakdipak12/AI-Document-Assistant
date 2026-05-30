from fastapi import FastAPI, Query

from app.routes.health import router as health_router
from app.routes.upload import router as upload_router
from app.routes.chat import router as chat_router


app = FastAPI(
    title="AI Support Intelligence Platform",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "message": "AI Support Intelligence Platform is running"
    }


@app.get("/evaluate")
def evaluate_system(
    temperature: float = Query(default=0.3, ge=0.0, le=1.0),
    max_tokens: int = Query(default=700, ge=100, le=1500),
    top_k: int = Query(default=10, ge=1, le=20),
    final_k: int = Query(default=3, ge=1, le=10)
):
    """
    Run custom evaluation without RAGAS.
    """

    try:
        from app.evaluation.evaluator import run_hybrid_evaluation

        return run_hybrid_evaluation(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            final_k=final_k
        )

    except Exception as error:
        return {
            "evaluation_type": "custom_no_ragas",
            "status": "evaluation_failed",
            "error": str(error),
            "message": "Evaluation failed, but upload and chat can still work."
        }
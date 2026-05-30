from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="demo")
    query: str

    # LLM controls
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=100, le=1500)

    # Retrieval controls
    top_k: int = Field(default=10, ge=1, le=20)
    final_k: int = Field(default=3, ge=1, le=10)
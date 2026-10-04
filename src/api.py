"""FastAPI application for MoMo RAG Q&A service."""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import list_available_configs
from src.pipeline import ask

app = FastAPI(
    title="MoMo RAG Q&A API",
    description="Hệ thống hỏi đáp tài liệu MoMo hỗ trợ so sánh hai cấu hình RAG A và B.",
    version="2.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của người dùng")
    config_id: str = Field("A", description="ID cấu hình pipeline ('A' hoặc 'B')")


class SourceItem(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    score: float
    text: str
    section: Optional[str] = None


class UsageMetadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LatencyMetadata(BaseModel):
    retrieval: float
    generation: float
    total: float


class AskResponse(BaseModel):
    question: str
    config_id: str
    answer: str
    sources: List[SourceItem]
    final_prompt: str
    usage: UsageMetadata
    cost_usd: float
    latency_ms: LatencyMetadata
    error: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/configs")
def get_configs() -> Dict[str, List[str]]:
    """Return list of available pipeline configuration IDs."""
    configs = list_available_configs()
    return {"configs": configs}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> Dict[str, Any]:
    """Execute RAG question answering pipeline on specified configuration."""
    available = list_available_configs()
    if req.config_id not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cấu hình '{req.config_id}' không tồn tại. "
                f"Danh sách cấu hình khả dụng: {available}"
            ),
        )

    result = ask(question=req.question, config_id=req.config_id)
    return result

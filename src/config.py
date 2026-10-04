"""Configuration loader and Pydantic validator for RAG MoMo."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Base directory is rag-momo root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MANUAL_DATA_DIR = DATA_DIR / "manual"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "analysis"


def get_api_key() -> str:
    """Retrieve Google API Key from environment.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set or empty.
    """
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise ValueError(
            "GOOGLE_API_KEY chưa được thiết lập hoặc chưa hợp lệ. "
            "Vui lòng cấu hình GOOGLE_API_KEY trong file .env"
        )
    return key


class ModelPricing(BaseModel):
    input: float = 0.0
    output: float = 0.0


class ModelsConfig(BaseModel):
    generator_model: str
    judge_model: str
    embedding_model: str
    pricing_usd_per_1m_tokens: Dict[str, ModelPricing] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    model: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 1024


class PipelineConfig(BaseModel):
    config_id: str
    description: Optional[str] = ""
    collection: str
    top_k: int = 3
    similarity_threshold: Optional[float] = None
    max_context_chars: int = 6000
    short_circuit_on_empty_context: bool = False
    prompt_file: str
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_models_cfg() -> ModelsConfig:
    """Load and validate configs/models.yaml using Pydantic.

    Raises:
        FileNotFoundError: If configs/models.yaml does not exist.
        ValueError: If any model is missing or still set to 'TODO_FILL'.
    """
    models_file = CONFIGS_DIR / "models.yaml"
    if not models_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {models_file}")

    with open(models_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    required_keys = ["generator_model", "judge_model", "embedding_model"]
    for key in required_keys:
        val = str(raw_data.get(key, "")).strip()
        if not val or "TODO_FILL" in val:
            raise ValueError(
                f"Giá trị '{key}' trong {models_file} chưa được cấu hình (đang là '{val}'). "
                f"Vui lòng chạy 'python -m src.list_models' để xem danh sách model Gemini hợp lệ "
                f"và điền tên model vào configs/models.yaml."
            )

    return ModelsConfig.model_validate(raw_data)


def load_pipeline_cfg(config_id: str) -> PipelineConfig:
    """Load pipeline config from configs/<config_id>.yaml and validate with Pydantic.

    Resolves llm.model to generator_model if llm.model is null/None.

    Args:
        config_id: 'A', 'B', etc.
    """
    cfg_file = CONFIGS_DIR / f"{config_id}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình pipeline: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    config = PipelineConfig.model_validate(raw_data)

    # Resolve model: nếu llm.model là null thì dùng generator_model
    if not config.llm.model:
        models_cfg = load_models_cfg()
        config.llm.model = models_cfg.generator_model

    return config


def list_available_configs() -> List[str]:
    """List all available pipeline configuration IDs in configs/ directory."""
    if not CONFIGS_DIR.exists():
        return []
    configs = []
    for f in sorted(CONFIGS_DIR.glob("*.yaml")):
        if f.stem != "models":
            configs.append(f.stem)
    return configs


# Backwards compatibility wrappers
def load_models_config() -> Dict[str, Any]:
    return load_models_cfg().model_dump()


def load_pipeline_config(config_id: str) -> Dict[str, Any]:
    return load_pipeline_cfg(config_id).model_dump()

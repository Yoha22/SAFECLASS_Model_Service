from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    MODEL_PATH: str = "safeclass_best.pt"
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_FPS: int = 15
    GDRIVE_FILE_ID: str = ""
    BACKEND_API_KEY: str = ""
    BACKEND_URL: str = "http://localhost:3000"
    PORT: int = 8001

    @field_validator("CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.50 <= v <= 0.95):
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0.50 and 0.95")
        return v

    @field_validator("MAX_FPS")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        if v < 1 or v > 60:
            raise ValueError("MAX_FPS must be between 1 and 60")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

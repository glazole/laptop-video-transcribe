"""Environment-backed application settings."""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = integer("APP_PORT", 7860)
    max_upload_mb: int = integer("MAX_UPLOAD_MB", 4096)
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", "./cache/whisper"))
    asr_model: str = os.getenv("ASR_MODEL", "large-v3")
    asr_device: str = os.getenv("ASR_DEVICE", "auto")
    asr_compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")
    asr_language: str = os.getenv("ASR_LANGUAGE", "ru")
    audio_chunk_seconds: int = integer("AUDIO_CHUNK_SECONDS", 900)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    ollama_timeout_seconds: int = integer("OLLAMA_TIMEOUT_SECONDS", 900)
    summary_max_chars: int = integer("SUMMARY_MAX_CHARS", 11000)
    summary_num_ctx: int = integer("SUMMARY_NUM_CTX", 8192)

    def validate(self) -> None:
        if self.audio_chunk_seconds < 30 or self.summary_max_chars < 1000:
            raise ValueError("AUDIO_CHUNK_SECONDS >= 30 and SUMMARY_MAX_CHARS >= 1000 required")
        if self.max_upload_mb < 1 or not 1 <= self.app_port <= 65535:
            raise ValueError("Invalid upload limit or port")
        if self.asr_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("ASR_DEVICE must be auto/cuda/cpu")

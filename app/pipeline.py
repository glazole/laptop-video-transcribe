"""Durable local job pipeline; ASR child process finishes before LLM begins."""
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import UUID, uuid4
from .config import Settings
from . import media
from .schemas import load_segments, as_transcript, srt_text, vtt_text
from .summary import Ollama, summarize
from .prompts import get_prompt
log = logging.getLogger(__name__)

def atomic_text(path: Path, content: str):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)

def job_path(cfg: Settings, job_id: str) -> Path:
    if str(UUID(job_id)) != job_id:
        raise ValueError("Некорректный ID")
    return cfg.data_dir.resolve() / "jobs" / job_id

def files_for_download(root: Path) -> list[str]:
    names = ("transcript.txt", "segments.json", "transcript.srt", "transcript.vtt")
    return [str(root / name) for name in names if (root / name).is_file()] + [
        str(p) for p in sorted(root.glob("summary_*.md"))]

def initial_job(input_path: str, cfg: Settings) -> tuple[str, Path]:
    source = Path(input_path)
    if not source.is_file():
        raise ValueError("Файл не найден")
    if source.suffix.lower() not in media.SUPPORTED:
        raise ValueError(f"Неподдерживаемый формат: {source.suffix}")
    if source.stat().st_size > cfg.max_upload_mb * 1024 * 1024:
        raise ValueError("Превышен лимит размера")
    job_id = str(uuid4())
    root = job_path(cfg, job_id)
    root.mkdir(parents=True, exist_ok=False)
    target = root / ("source" + source.suffix.lower())
    shutil.copyfile(source, target)
    atomic_text(root / "job.json", json.dumps({"job_id": job_id, "filename": source.name,
        "source": target.name, "status": "created"}, ensure_ascii=False, indent=2))
    return job_id, root

def set_status(root: Path, status: str, **kwargs):
    path = root / "job.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(status=status, **kwargs)
    atomic_text(path, json.dumps(state, ensure_ascii=False, indent=2))

def transcribe_job(root: Path, model: str, cfg: Settings, progress=None) -> str:
    target = root / "segments.json"
    if target.is_file():
        if json.loads(target.read_text(encoding="utf-8")).get("model") != model:
            raise ValueError("У задания другая модель ASR")
        return as_transcript(load_segments(target))
    record = json.loads((root / "job.json").read_text(encoding="utf-8"))
    source = root / record["source"]
    if progress:
        progress("FFprobe: проверка аудиодорожки")
    info = media.probe(source)
    set_status(root, "extracting", duration_seconds=float(info.get("format", {}).get("duration") or 0), asr_model=model)
    wav = root / "audio_16k.wav"
    if not wav.is_file():
        if progress:
            progress("FFmpeg: извлечение аудио")
        media.extract_audio(source, wav)
    if progress:
        progress("FFmpeg: нарезка аудио")
    chunks = media.chunks_from_wav(wav, root / "chunks", cfg.audio_chunk_seconds)
    manifest = root / "chunks.json"
    atomic_text(manifest, json.dumps([{"path": str(path.resolve()), "offset": offset}
                                     for path, offset in chunks]))
    set_status(root, "transcribing", chunks=len(chunks))
    if progress:
        progress(f"faster-whisper: {len(chunks)} аудиофрагментов")
    cmd = [sys.executable, "-m", "app.asr", "--chunks-manifest", str(manifest.resolve()),
           "--output", str(target.resolve()), "--model", model]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        log.error("ASR failed: %s", proc.stderr[-8000:])
        raise RuntimeError(f"Ошибка ASR: {proc.stderr[-2000:]}")
    segments = load_segments(target)
    text = as_transcript(segments)
    atomic_text(root / "transcript.txt", text)
    atomic_text(root / "transcript.srt", srt_text(segments))
    atomic_text(root / "transcript.vtt", vtt_text(segments))
    wav.unlink(missing_ok=True)
    shutil.rmtree(root / "chunks", ignore_errors=True)
    manifest.unlink(missing_ok=True)
    set_status(root, "transcribed", segments=len(segments))
    return text

def summarize_job(root: Path, kind: str, cfg: Settings, progress=None) -> str:
    get_prompt(kind)
    segments = load_segments(root / "segments.json")
    set_status(root, "summarizing", summary_kind=kind)
    client = Ollama(cfg.ollama_url, cfg.ollama_model,
                    cfg.ollama_timeout_seconds, cfg.summary_num_ctx)
    summary = summarize(segments, kind, client, max_chars=cfg.summary_max_chars, progress=progress)
    atomic_text(root / f"summary_{kind}.md", summary)
    set_status(root, "completed")
    return summary

def process(input_path: str, model: str, kind: str, cfg: Settings, progress=None):
    job_id, root = initial_job(input_path, cfg)
    try:
        transcript = transcribe_job(root, model, cfg, progress)
    except Exception as exc:
        set_status(root, "asr_failed", error=str(exc))
        raise
    try:
        summary = summarize_job(root, kind, cfg, progress)
        status = "Готово: транскрипция и резюме сохранены."
    except Exception as exc:
        log.exception("Summary failed; transcript retained")
        set_status(root, "summary_failed", error=str(exc))
        summary = f"Саммаризация не выполнена: {exc}. Используйте повторную саммаризацию."
        status = "Транскрипция готова; саммаризацию можно повторить."
    return job_id, transcript, summary, files_for_download(root), status

def rerun_summary(job_id: str, kind: str, cfg: Settings, progress=None):
    root = job_path(cfg, job_id)
    if not (root / "segments.json").is_file():
        raise ValueError("Не найдена транскрипция для этого задания")
    summary = summarize_job(root, kind, cfg, progress)
    return summary, files_for_download(root), "Саммаризация выполнена без повторного ASR."

"""FFmpeg input inspection and bounded audio chunks."""
import json
from pathlib import Path
import subprocess

SUPPORTED = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

def run_command(args: list[str], timeout: int = 7200) -> str:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Не найдена утилита: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Время ожидания {args[0]} истекло") from exc
    if result.returncode:
        raise RuntimeError(f"{args[0]}: {result.stderr[-1600:]}")
    return result.stdout

def probe(path: Path) -> dict:
    info = json.loads(run_command(["ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type", "-of", "json", str(path)]))
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise ValueError("В файле отсутствует аудиодорожка")
    return info

def extract_audio(source: Path, wav: Path) -> None:
    run_command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(wav)])
    if not wav.is_file() or wav.stat().st_size < 44:
        raise RuntimeError("FFmpeg не создал аудиофайл")

def chunks_from_wav(wav: Path, out_dir: Path, seconds: int) -> list[tuple[Path, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(wav), "-f", "segment", "-segment_time", str(seconds), "-c:a", "copy",
        str(out_dir / "chunk_%05d.wav")])
    chunks = []
    offset = 0.0
    for path in sorted(out_dir.glob("chunk_*.wav")):
        duration = float(probe(path).get("format", {}).get("duration", 0))
        if duration <= 0:
            raise ValueError(f"Нулевая длительность: {path}")
        chunks.append((path, offset))
        offset += duration
    if not chunks:
        raise RuntimeError("Аудиофрагменты не созданы")
    return chunks

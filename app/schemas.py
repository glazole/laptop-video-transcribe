"""Structured transcript, diarization-ready."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError("Invalid timestamps")

def clock(seconds: float, *, ms: bool = False, srt: bool = False) -> str:
    ms_total = max(0, round(seconds * 1000))
    h, rest = divmod(ms_total, 3600000)
    m, rest = divmod(rest, 60000)
    s, millis = divmod(rest, 1000)
    if ms:
        return f"{h:02d}:{m:02d}:{s:02d}{',' if srt else '.'}{millis:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

def segment_line(s: Segment) -> str:
    label = f"{s.speaker}: " if s.speaker else ""
    return f"[{clock(s.start)}] {label}{s.text.strip()}"

def as_transcript(segments: list[Segment]) -> str:
    return "\n".join(segment_line(s) for s in segments if s.text.strip())

def save_segments(path: Path, segments: list[Segment], *, model: str, language: str) -> None:
    data = {"schema_version": 1, "model": model, "language": language,
            "segments": [asdict(s) for s in segments]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_segments(path: Path) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported schema")
    return [Segment(**item) for item in data["segments"]]

def srt_text(segments: list[Segment]) -> str:
    output = []
    for n, s in enumerate((s for s in segments if s.text.strip()), 1):
        speaker = f"{s.speaker}: " if s.speaker else ""
        output.append(f"{n}\n{clock(s.start, ms=True, srt=True)} --> {clock(s.end, ms=True, srt=True)}\n{speaker}{s.text.strip()}\n")
    return "\n".join(output)

def vtt_text(segments: list[Segment]) -> str:
    output = ["WEBVTT", ""]
    for s in segments:
        if s.text.strip():
            speaker = f"{s.speaker}: " if s.speaker else ""
            output.append(f"{clock(s.start, ms=True)} --> {clock(s.end, ms=True)}\n{speaker}{s.text.strip()}\n")
    return "\n".join(output)

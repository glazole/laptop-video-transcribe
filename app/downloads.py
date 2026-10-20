"""Stage only generated result files under system temp for Gradio exports.

Never grant Gradio access to the entire persistent jobs directory: it may
contain uploaded videos and other data not intended as downloadable results.
"""
from pathlib import Path
from shutil import copy2
from tempfile import mkdtemp

_ALLOWED_NAMES = {"transcript.txt", "transcript.srt", "transcript.vtt", "segments.json"}


def stage_downloads(files: list[str]) -> list[str]:
    if not files:
        return []
    output_dir = Path(mkdtemp(prefix="lvt-gradio-export-"))
    result = []
    for source in files:
        src = Path(source)
        if src.name not in _ALLOWED_NAMES and not (src.name.startswith("summary_") and src.suffix == ".md"):
            raise ValueError(f"Недопустимый файл для выдачи: {src.name}")
        if not src.is_file():
            raise FileNotFoundError(f"Не найден результат: {src}")
        dst = output_dir / src.name
        copy2(src, dst)
        result.append(str(dst))
    return result

"""Isolated ASR subprocess; exits to reclaim all GPU memory."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
from .config import Settings
from .schemas import Segment, save_segments
logger = logging.getLogger(__name__)

def recognize(chunks: list[tuple[Path, float]], *, model_name: str, language: str,
              cache_dir: Path, device: str, compute_type: str) -> list[Segment]:
    from faster_whisper import WhisperModel
    devices = [device] if device != "auto" else ["cuda", "cpu"]
    last_error = None
    for current_device in devices:
        model = None
        output = []
        try:
            mode = compute_type if current_device == "cuda" else "int8"
            logger.info("ASR %s on %s (%s)", model_name, current_device, mode)
            model = WhisperModel(model_name, device=current_device, compute_type=mode,
                                 download_root=str(cache_dir))
            for index, (path, offset) in enumerate(chunks, 1):
                logger.info("ASR chunk %d/%d", index, len(chunks))
                iterator, _ = model.transcribe(str(path), language=language or None, beam_size=5,
                                               vad_filter=True, condition_on_previous_text=False)
                for s in iterator:
                    if s.text.strip():
                        output.append(Segment(max(0., offset + s.start), max(0., offset + s.end),
                                              s.text.strip()))
            return output
        except Exception as exc:
            last_error = exc
            if current_device != "cuda" or device == "cuda":
                raise
            logger.warning("CUDA failure, retry on CPU: %s", exc)
        finally:
            del model
            gc.collect()
    raise RuntimeError(f"ASR failed: {last_error}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = Settings()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.chunks_manifest).read_text(encoding="utf-8"))
    segments = recognize([(Path(x["path"]), float(x["offset"])) for x in entries],
        model_name=args.model, language=cfg.asr_language, cache_dir=cfg.cache_dir.resolve(),
        device=cfg.asr_device, compute_type=cfg.asr_compute_type)
    target = Path(args.output)
    temp = target.with_suffix(".tmp")
    save_segments(temp, segments, model=args.model, language=cfg.asr_language)
    os.replace(temp, target)

if __name__ == "__main__":
    main()

import json
import pytest
from app.config import Settings
from app.pipeline import job_path, initial_job, files_for_download, rerun_summary
from app.schemas import Segment, save_segments

def test_path_restriction(tmp_path):
    with pytest.raises(ValueError):
        job_path(Settings(data_dir=tmp_path), "../other")

def test_upload(tmp_path):
    source = tmp_path / "video.MKV"
    source.write_bytes(b"fake")
    cfg = Settings(data_dir=tmp_path, max_upload_mb=1)
    job_id, root = initial_job(str(source), cfg)
    assert (root / "source.mkv").read_bytes() == b"fake"
    assert job_path(cfg, job_id) == root
    assert json.loads((root / "job.json").read_text())["status"] == "created"
    assert files_for_download(root) == []

def test_rerun_without_asr(tmp_path, monkeypatch):
    cfg = Settings(data_dir=tmp_path)
    source = tmp_path / "sample.wav"
    source.write_bytes(b"fake")
    job_id, root = initial_job(str(source), cfg)
    save_segments(root / "segments.json", [Segment(1, 2, "Текст")],
                  model="small", language="ru")
    class FakeClient:
        def __init__(self, *args): pass
        def chat(self, system, user, max_tokens=1100): return "Резюме"
        def unload(self): pass
    monkeypatch.setattr("app.pipeline.Ollama", FakeClient)
    summary, files, status = rerun_summary(job_id, "general", cfg)
    assert summary == "Резюме"
    assert any(f.endswith("summary_general.md") for f in files)

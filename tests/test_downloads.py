import pytest
from pathlib import Path
from app.downloads import stage_downloads


def test_stage_results_under_temp(tmp_path):
    original = tmp_path / "transcript.txt"
    original.write_text("Тест транскрибации", encoding="utf-8")
    result = stage_downloads([str(original)])
    assert len(result) == 1
    assert Path(result[0]).read_text(encoding="utf-8") == "Тест транскрибации"
    assert Path(result[0]).resolve() != original.resolve()
    assert "lvt-gradio-export-" in result[0]


def test_reject_uploads(tmp_path):
    original = tmp_path / "source.mkv"
    original.write_text("secret")
    with pytest.raises(ValueError):
        stage_downloads([str(original)])


def test_empty():
    assert stage_downloads([]) == []

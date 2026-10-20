import pytest
from app.schemas import Segment, as_transcript, clock, srt_text, vtt_text, save_segments, load_segments

def test_export(tmp_path):
    segments = [Segment(0.5, 1.5, "Привет"), Segment(3600, 3602, "Вопрос", "Спикер 1")]
    target = tmp_path / "segments.json"
    save_segments(target, segments, model="turbo", language="ru")
    assert load_segments(target) == segments
    assert "[01:00:00] Спикер 1: Вопрос" in as_transcript(segments)
    assert "00:00:00,500 --> 00:00:01,500" in srt_text(segments)
    assert "WEBVTT" in vtt_text(segments)
    assert clock(1.239, ms=True) == "00:00:01.239"

def test_invalid_time():
    with pytest.raises(ValueError):
        Segment(2, 1, "bad")

from app.schemas import Segment
from app.summary import make_blocks, pack_notes, summarize

class FakeLLM:
    def __init__(self):
        self.calls = []
        self.unloaded = False
    def chat(self, system, user, max_tokens=1100):
        self.calls.append((system, user, max_tokens))
        return "Резюме [00:00:00]: " + str(len(self.calls))
    def unload(self):
        self.unloaded = True

def test_no_dropped_segments():
    segments = [Segment(i * 3, i * 3 + 2, f"Предложение номер {i}") for i in range(40)]
    blocks = make_blocks(segments, 180)
    assert len(blocks) > 1
    for segment in segments:
        assert segment.text in "\n".join(blocks)

def test_summary_system_prompts():
    llm = FakeLLM()
    segments = [Segment(i, i + 1, f"Фрагмент номер {i}") for i in range(45)]
    assert summarize(segments, "meeting", llm, max_chars=1000).startswith("Резюме")
    assert llm.unloaded
    assert any("ЗАДАЧА MAP" in item[0] for item in llm.calls)
    assert any("ЗАДАЧА REDUCE" in item[0] for item in llm.calls)

def test_pack():
    assert pack_notes(["aaa", "bbb"], 40) == ["Часть 1:\naaa\n\nЧасть 2:\nbbb"]

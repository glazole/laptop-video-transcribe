import pytest
from app.prompts import PROMPTS, get_prompt

def test_prompts_per_mode():
    assert set(PROMPTS) == {"general", "meeting", "lecture", "interview", "actions"}
    for prompt in PROMPTS.values():
        assert "ЗАДАЧА MAP" in prompt.map_system
        assert "ЗАДАЧА REDUCE" in prompt.reduce_system
        assert "Не придумывай" in prompt.map_system

def test_invalid_mode():
    with pytest.raises(ValueError):
        get_prompt("bad")

"""Hierarchical map/reduce Qwen summarization via Ollama."""
import logging
import httpx
from .prompts import get_prompt
from .schemas import Segment, segment_line
log = logging.getLogger(__name__)

def make_blocks(segments: list[Segment], max_chars: int) -> list[str]:
    blocks, current, length = [], [], 0
    for seg in segments:
        line = segment_line(seg)
        if len(line) > max_chars:
            words, pieces, buf = line.split(), [], ""
            for word in words:
                if buf and len(buf) + len(word) + 1 > max_chars:
                    pieces.append(buf)
                    buf = word
                else:
                    buf = f"{buf} {word}".strip()
            if buf:
                pieces.append(buf)
        else:
            pieces = [line]
        for piece in pieces:
            if current and length + len(piece) + 1 > max_chars:
                blocks.append("\n".join(current))
                current, length = [], 0
            current.append(piece)
            length += len(piece) + 1
    if current:
        blocks.append("\n".join(current))
    return blocks

def pack_notes(notes: list[str], max_chars: int) -> list[str]:
    groups, current = [], ""
    for i, note in enumerate(notes, 1):
        piece = f"Часть {i}:\n{note.strip()}"
        if len(piece) > max_chars:
            raise ValueError("Промежуточное резюме превышает лимит блока")
        if current and len(current) + len(piece) + 2 > max_chars:
            groups.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        groups.append(current)
    return groups

class Ollama:
    def __init__(self, url: str, model: str, timeout: int, num_ctx: int):
        self.url, self.model, self.timeout, self.num_ctx = url, model, timeout, num_ctx
        self._model_was_used = False

    def chat(self, system: str, user: str, *, max_tokens: int = 1100) -> str:
        payload = {"model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.1, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            "keep_alive": "10m"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.url}/api/chat", json=payload)
                response.raise_for_status()
                answer = response.json().get("message", {}).get("content", "").strip()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Модель {self.model} не установлена в Ollama. "
                    f"Выполните: docker compose exec ollama ollama pull {self.model}; "
                    f"затем проверьте: docker compose exec ollama ollama list"
                ) from exc
            raise RuntimeError(f"Ошибка Ollama ({self.model}): {exc}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"Ollama недоступна или модель отсутствует ({self.model}): {exc}") from exc
        if not answer:
            raise RuntimeError("Ollama вернула пустой ответ")
        self._model_was_used = True
        return answer

    def unload(self):
        if not self._model_was_used:
            return
        try:
            with httpx.Client(timeout=15) as client:
                client.post(f"{self.url}/api/generate", json={"model": self.model,
                                                               "prompt": "", "keep_alive": 0})
        except Exception as exc:
            log.warning("Model unload failed: %s", exc)

def summarize(segments: list[Segment], kind: str, client: Ollama, max_chars=11000, progress=None) -> str:
    if not segments:
        raise ValueError("Нет распознанной речи")
    prompt = get_prompt(kind)
    blocks = make_blocks(segments, max_chars)
    try:
        notes = []
        for i, block in enumerate(blocks, 1):
            if progress:
                progress(f"Саммаризация блока {i}/{len(blocks)}")
            notes.append(client.chat(prompt.map_system,
                f"Фрагмент {i}/{len(blocks)}:\n{block}", max_tokens=850))
        rounds = 0
        while len(notes) > 1:
            rounds += 1
            if rounds > 30:
                raise RuntimeError("Слишком много уровней свёртки")
            groups = pack_notes(notes, max_chars)
            if len(groups) >= len(notes):
                raise RuntimeError("Резюме не помещаются в контекст: уменьшите размер выходных блоков")
            if progress:
                progress(f"Объединение резюме: {len(groups)} групп")
            notes = [client.chat(prompt.reduce_system, f"Объедини резюме:\n{x}",
                    max_tokens=1250) for x in groups]
        if progress:
            progress("Итоговое резюме")
        return client.chat(prompt.reduce_system, f"Составь итоговый документ:\n{notes[0]}",
                           max_tokens=1500)
    finally:
        client.unload()

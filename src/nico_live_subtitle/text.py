from __future__ import annotations

from difflib import SequenceMatcher


_SENTENCE_ENDINGS = frozenset("。！？!?")
_CLOSING_MARKS = frozenset("」』）)]】〉》\"'”’")


def split_japanese_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(stripped):
        if stripped[index] not in _SENTENCE_ENDINGS:
            index += 1
            continue
        end = index + 1
        while end < len(stripped) and stripped[end] in _CLOSING_MARKS:
            end += 1
        parts.append(stripped[start:end].strip())
        start = end
        index = end
    tail = stripped[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


class TranscriptStabilizer:
    """只提交增量识别中已经完整结束、且不会再变化的句子。"""

    def __init__(self) -> None:
        self._committed = ""

    def push(self, text: str, is_final: bool) -> tuple[list[str], str]:
        remaining = self._strip_committed_overlap(text.strip())
        if not remaining:
            return [], ""
        parts = split_japanese_sentences(remaining)
        if is_final:
            complete = parts
            pending = ""
        elif len(parts) > 1:
            complete = parts[:-1]
            pending = parts[-1]
        else:
            complete = []
            pending = remaining
        if complete:
            self._committed += "".join(complete)
        return complete, pending

    def _strip_committed_overlap(self, text: str) -> str:
        if not self._committed:
            return text
        if text.startswith(self._committed):
            return text[len(self._committed) :].strip()
        tail = self._committed[-100:]
        max_overlap = min(len(tail), len(text))
        for length in range(max_overlap, 2, -1):
            if text.startswith(tail[-length:]):
                return text[length:].strip()
        prefix = text[: len(tail)]
        if len(prefix) >= 6 and SequenceMatcher(
            None, tail, prefix, autojunk=False
        ).ratio() >= 0.86:
            return text[len(prefix) :].lstrip("。！？!?").strip()
        return text

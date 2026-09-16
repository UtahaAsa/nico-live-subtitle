from __future__ import annotations

import ctypes
import re
import sys
from dataclasses import dataclass
from typing import Sequence

from .config import LexiconConfig
from .lexicon import AnimeLexicon


_LIVE_TITLE_HINTS = (
    "生放送",
    "ライブ配信",
    "配信中",
    "livestream",
    "live stream",
    "streaming",
    "twitch",
)
_ANIME_TITLE_HINTS = (
    "アニメ",
    "anime",
    "episode",
    "season",
    "シーズン",
    "ニコニコ動画",
    "dアニメストア",
    "crunchyroll",
)


@dataclass(frozen=True)
class SourceContext:
    kind: str
    profile: str
    label: str


def detect_source_context(
    config: LexiconConfig,
    catalog: Sequence[AnimeLexicon],
    window_titles: Sequence[str] | None = None,
) -> SourceContext:
    if config.profile == "live":
        return SourceContext("live", "", "日英直播")
    if config.profile not in {"auto", ""}:
        profile = next((item for item in catalog if item.id == config.profile), None)
        if profile is None:
            raise ValueError(f"找不到作品词库：{config.profile}")
        return SourceContext("anime", profile.id, profile.title)
    if config.profile == "":
        return SourceContext("anime", "", "通用动画")

    titles = list(window_titles) if window_titles is not None else list_window_titles()
    best: tuple[int, AnimeLexicon] | None = None
    for lexicon in catalog:
        if lexicon.id in {"anime-common", "live-common"}:
            continue
        candidates = (lexicon.title, *lexicon.aliases)
        for candidate in candidates:
            normalized_candidate = _normalize(candidate)
            if len(normalized_candidate) < 3:
                continue
            for title in titles:
                if normalized_candidate in _normalize(title):
                    score = len(normalized_candidate)
                    if best is None or score > best[0]:
                        best = (score, lexicon)
    if best is not None:
        return SourceContext("anime", best[1].id, best[1].title)

    normalized_titles = tuple(_normalize(title) for title in titles)
    if any(
        _normalize(hint) in title
        for hint in _LIVE_TITLE_HINTS
        for title in normalized_titles
    ):
        return SourceContext("live", "", "日英直播")
    if any(
        _normalize(hint) in title
        for hint in _ANIME_TITLE_HINTS
        for title in normalized_titles
    ) or any(re.search(r"第\d{1,3}話", title) for title in titles):
        return SourceContext("anime", "", "通用动画")
    return SourceContext("live", "", "日英直播")


def list_window_titles() -> list[str]:
    if sys.platform != "win32":
        return []
    user32 = ctypes.windll.user32
    titles: list[str] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def collect(window: int, _: int) -> bool:
        if not user32.IsWindowVisible(window):
            return True
        length = user32.GetWindowTextLengthW(window)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window, buffer, length + 1)
        title = buffer.value.strip()
        if title:
            titles.append(title)
        return True

    callback = callback_type(collect)
    user32.EnumWindows(callback, 0)
    return titles


def _normalize(text: str) -> str:
    return re.sub(r"[\s:：・·\-—_]+", "", text).casefold()

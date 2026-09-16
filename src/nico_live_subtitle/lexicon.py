from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .config import LexiconConfig


@dataclass(frozen=True)
class LexiconTerm:
    sources: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class AnimeLexicon:
    id: str
    title: str
    aliases: tuple[str, ...]
    hotwords: tuple[str, ...]
    terms: tuple[LexiconTerm, ...]


@dataclass(frozen=True)
class LexiconBundle:
    title: str
    hotwords: str
    glossary: str
    term_count: int


def load_lexicon_catalog(directory: str | Path) -> list[AnimeLexicon]:
    root = Path(directory).resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"作品词库路径不是目录：{root}")

    lexicons: list[AnimeLexicon] = []
    seen_ids: set[str] = set()
    for path in sorted(root.glob("*.json")):
        lexicon = _load_lexicon(path)
        if lexicon.id in seen_ids:
            raise ValueError(f"作品词库 ID 重复：{lexicon.id}")
        seen_ids.add(lexicon.id)
        lexicons.append(lexicon)
    return sorted(lexicons, key=lambda item: item.title.casefold())


def build_lexicon_bundle(
    config: LexiconConfig,
    custom_hotwords: str = "",
    custom_glossary: str = "",
    base_lexicon_id: str = "anime-common",
) -> LexiconBundle:
    catalog = {item.id: item for item in load_lexicon_catalog(config.directory)}
    selected: list[AnimeLexicon] = []
    if config.include_common and base_lexicon_id in catalog:
        selected.append(catalog[base_lexicon_id])
    if config.profile:
        profile = catalog.get(config.profile)
        if profile is None:
            raise ValueError(f"找不到作品词库：{config.profile}")
        if profile.id != base_lexicon_id:
            selected.append(profile)

    hotwords: dict[str, None] = {}
    terms: dict[str, str] = {}
    for lexicon in selected:
        for item in (*lexicon.aliases, *lexicon.hotwords):
            if item.strip():
                hotwords[item.strip()] = None
        for term in lexicon.terms:
            for source in term.sources:
                hotwords[source] = None
                terms[source] = term.target

    for item in re.split(r"[,，\s]+", custom_hotwords.strip()):
        if item:
            hotwords[item] = None
    for source, target in _parse_custom_glossary(custom_glossary):
        hotwords[source] = None
        terms[source] = target

    profile_title = next(
        (item.title for item in selected if item.id != base_lexicon_id),
        "通用直播" if base_lexicon_id == "live-common" else "通用动画",
    )
    return LexiconBundle(
        title=profile_title,
        hotwords=" ".join(hotwords),
        glossary=", ".join(f"{source}={target}" for source, target in terms.items()),
        term_count=len(terms),
    )


def _load_lexicon(path: Path) -> AnimeLexicon:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取作品词库 {path.name}：{error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"作品词库 {path.name} 顶层必须是对象")
    if raw.get("schema_version") != 1:
        raise ValueError(f"作品词库 {path.name} 的 schema_version 必须为 1")

    lexicon_id = _required_text(raw, "id", path)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", lexicon_id):
        raise ValueError(f"作品词库 {path.name} 的 id 格式无效")
    title = _required_text(raw, "title", path)
    aliases = _text_list(raw.get("aliases", []), "aliases", path)
    hotwords = _text_list(raw.get("hotwords", []), "hotwords", path)

    raw_terms = raw.get("terms", [])
    if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
        raise ValueError(f"作品词库 {path.name} 的 terms 必须是数组")
    terms: list[LexiconTerm] = []
    for index, raw_term in enumerate(raw_terms):
        if not isinstance(raw_term, Mapping):
            raise ValueError(f"作品词库 {path.name} 的 terms[{index}] 必须是对象")
        source_value = raw_term.get("source")
        if isinstance(source_value, str):
            sources = (source_value.strip(),)
        else:
            sources = _text_list(source_value, f"terms[{index}].source", path)
        sources = tuple(item for item in sources if item)
        target = str(raw_term.get("target", "")).strip()
        if not sources or not target:
            raise ValueError(
                f"作品词库 {path.name} 的 terms[{index}] 缺少 source 或 target"
            )
        terms.append(LexiconTerm(sources, target))
    return AnimeLexicon(lexicon_id, title, aliases, hotwords, tuple(terms))


def _required_text(raw: Mapping[object, object], key: str, path: Path) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"作品词库 {path.name} 缺少 {key}")
    return value


def _text_list(value: object, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"作品词库 {path.name} 的 {field} 必须是字符串数组")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(value):
        raise ValueError(f"作品词库 {path.name} 的 {field} 不能包含空值")
    return result


def _parse_custom_glossary(glossary: str) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    for item in glossary.replace("，", ",").split(","):
        separator = "=" if "=" in item else "＝" if "＝" in item else None
        if separator is None:
            continue
        source, target = (part.strip() for part in item.split(separator, 1))
        if source and target:
            terms.append((source, target))
    return terms

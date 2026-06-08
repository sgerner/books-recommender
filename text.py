from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_-]{1,}")
_FRONTMATTER_RE = re.compile(r'^---\s*$', re.M)


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return ''.join(self.parts)


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ''
    stripper = _HTMLStripper()
    stripper.feed(html.unescape(raw))
    text = stripper.text()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [t.lower().replace('’', "'") for t in _TOKEN_RE.findall(text or '')]


def parse_scalar(value: str) -> Any:
    s = value.strip()
    if s == '':
        return ''
    low = s.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'null', 'none', '~'}:
        return None
    if s and s[0] in '"\'[{(':
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    for caster in (int, float):
        try:
            return caster(s)
        except Exception:
            pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith('---'):
        return {}, text
    parts = _FRONTMATTER_RE.split(text, maxsplit=2)
    if len(parts) < 3:
        return {}, text
    _, frontmatter, body = parts[0], parts[1], parts[2]
    meta: dict[str, Any] = {}
    for raw_line in frontmatter.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        meta[key.strip()] = parse_scalar(value)
    return meta, body.lstrip('\n')


def extract_heading_section(body: str, heading: str) -> str:
    pattern = re.compile(rf'^##\s+{re.escape(heading)}\s*$', re.I | re.M)
    m = pattern.search(body)
    if not m:
        return ''
    start = m.end()
    tail = body[start:]
    next_heading = re.search(r'^##\s+.+$', tail, re.M)
    return tail[: next_heading.start()] if next_heading else tail


def extract_book_title(body: str) -> str:
    m = re.search(r'^#\s+(.+)$', body, re.M)
    return m.group(1).strip() if m else ''


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [x.strip() for x in re.split(r'[,;/]', value) if x.strip()]
    return [str(value).strip()]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


def best_book_uid(meta: dict[str, Any], path: Path) -> str:
    for key in ('goodreads_id', 'storygraph_id', 'isbn', 'isbn13'):
        if meta.get(key):
            return f"{key}:{meta[key]}"
    return f"path:{path.as_posix()}"


def parse_note(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding='utf-8', errors='replace')
    meta, body = parse_frontmatter(raw)
    title = meta.get('title') or extract_book_title(body) or path.stem
    author = meta.get('author') or ''
    summary = extract_heading_section(body, 'Summary').strip()
    review = extract_heading_section(body, 'Review').strip()
    def _s(value: Any) -> str:
        return '' if value is None else str(value).strip()
    return {
        'path': str(path),
        'title': title,
        'author': author,
        'rating': meta.get('rating'),
        'goodreads_id': _s(meta.get('goodreads_id')),
        'storygraph_id': _s(meta.get('storygraph_id')),
        'isbn': _s(meta.get('isbn') or meta.get('isbn13')),
        'tags': listify(meta.get('tags')),
        'themes': listify(meta.get('themes')),
        'summary': html_to_text(summary),
        'review': html_to_text(review),
        'body': body,
        'meta': meta,
        'uid': best_book_uid(meta, path),
    }

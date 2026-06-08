from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from urllib.parse import urlparse

from .text import slugify

_URL_RE = re.compile(r'https?://[^\s<>\)\]]+')


def _read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ''
    return p.read_text(encoding='utf-8', errors='replace')


def extract_section(text: str, heading: str) -> str:
    pattern = re.compile(rf'^##\s+{re.escape(heading)}\s*$', re.M)
    m = pattern.search(text)
    if not m:
        return ''
    start = m.end()
    tail = text[start:]
    next_heading = re.search(r'^##\s+.+$', tail, re.M)
    return tail[: next_heading.start()] if next_heading else tail


def extract_urls(text: str) -> list[str]:
    urls = []
    seen = set()
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(').,;')
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def source_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(':', 1)[0]
    path = parsed.path.strip('/')
    if not path:
        return slugify(host)
    short = path.replace('/', '-')
    if parsed.query:
        short = f'{short}'
    return slugify(f'{host}-{short}')


def classify_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    if 'goodreads.com' in host and ('rss' in path or 'review/list_rss' in path):
        return 'goodreads-rss'
    if 'nytimes.com' in host:
        return 'nyt-books'
    if url.lower().endswith('.xml') or url.lower().endswith('.rss') or 'feed' in path:
        return 'rss'
    return 'web'


def default_weight(kind: str) -> float:
    return {
        'goodreads-rss': 0.35,
        'nyt-books': 0.55,
        'rss': 0.4,
        'web': 0.45,
    }.get(kind, 0.4)


def load_inbox_sources(vault_path: str, inbox_relpath: str = 'Media/Book Sources Inbox.md') -> list[dict[str, Any]]:
    vault = Path(vault_path)
    inbox = Path(inbox_relpath)
    if inbox.is_absolute():
        inbox_path = inbox
    elif inbox.parts and inbox.parts[0] == 'Media':
        inbox_path = vault.parent / Path(*inbox.parts[1:])
    else:
        inbox_path = vault / inbox
    text = _read_text(inbox_path)
    if not text:
        return []
    section = extract_section(text, '🆕 New Sources') or text
    urls = extract_urls(section)
    sources: list[dict[str, Any]] = []
    for url in urls:
        kind = classify_source(url)
        sources.append({
            'name': source_name_from_url(url),
            'kind': kind,
            'url': url,
            'enabled': True,
            'weight': default_weight(kind),
            'config': {},
            'origin': str(inbox_path),
        })
    return sources


def combined_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sources = list(cfg.get('sources', []))
    inbox_path = cfg.get('sources_inbox_path')
    if cfg.get('vault_path') and inbox_path:
        sources.extend(load_inbox_sources(cfg['vault_path'], inbox_path))
    # de-dupe by name/url while preserving order
    out = []
    seen = set()
    for src in sources:
        key = (src.get('name') or '', src.get('url') or '')
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def archive_sources_in_inbox(inbox_path: str | Path, urls: list[str]) -> dict[str, int]:
    path = Path(inbox_path)
    text = _read_text(path)
    if not text:
        return {'archived': 0, 'removed': 0}

    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        return {'archived': 0, 'removed': 0}

    new_match = re.search(rf'(?m)^##\s+{re.escape("🆕 New Sources")}\s*$', text)
    if not new_match:
        return {'archived': 0, 'removed': 0}
    processed_match = re.search(rf'(?m)^##\s+{re.escape("✅ Processed")}\s*$', text)

    new_start = new_match.end()
    new_end = processed_match.start() if processed_match else len(text)
    new_header = text[:new_start].rstrip()
    new_section = text[new_start:new_end]
    processed_section = text[processed_match.end():] if processed_match else ''

    removed = 0
    kept_lines: list[str] = []
    for line in new_section.splitlines():
        if any(url in line for url in urls):
            removed += 1
            continue
        kept_lines.append(line)
    while kept_lines and not kept_lines[0].strip():
        kept_lines.pop(0)
    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()

    today = datetime.now(timezone.utc).date().isoformat()
    archived_lines = [f'- <{url}> — archived {today}' for url in urls if url not in processed_section]

    if not archived_lines and removed == 0:
        return {'archived': 0, 'removed': 0}

    rebuilt = [new_header, '']
    if kept_lines:
        rebuilt.append('\n'.join(kept_lines))
        rebuilt.append('')
    if processed_match:
        rebuilt.append('## ✅ Processed')
        rebuilt.append('')
        processed_body = processed_section.strip('\n')
        if processed_body:
            rebuilt.append(processed_body.rstrip())
            if archived_lines:
                rebuilt.append('')
        if archived_lines:
            rebuilt.append('\n'.join(archived_lines))
        rebuilt.append('')
    else:
        rebuilt.append('## ✅ Processed')
        rebuilt.append('')
        rebuilt.append('\n'.join(archived_lines))
        rebuilt.append('')

    path.write_text('\n'.join(rebuilt).rstrip() + '\n', encoding='utf-8')
    return {'archived': len(archived_lines), 'removed': removed}

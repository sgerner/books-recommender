from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable

from .text import html_to_text

BOOK_URL_RE = re.compile(r"https?://www\.goodreads\.com/book/show/(\d+)[^\"'\s<>]*", re.I)
ALT_RE = re.compile(r"alt=[\"']([^\"']+)[\"']", re.I)
HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.I)
AUTHOR_RE = re.compile(r"author:\s*(.+?)(?:\s{2,}|\s+rating:|$)", re.I | re.S)


def fetch_url(url: str, timeout: int = 30) -> tuple[bytes, str | None]:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 Hermes Book Recommender'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get_content_type()


# ---------------------------------------------------------------------------
# XML sanitization for feeds that aren't well-formed (Goodreads, etc.)
# ---------------------------------------------------------------------------
# Invalid XML 1.0 characters: anything below 0x20 except 0x09 (tab), 0x0A
# (LF), 0x0D (CR), plus surrogate blocks (which valid UTF-8 won't contain
# anyway, but we strip them just in case).
_INVALID_XML_CHARS_RE = re.compile(
    '[\x00-\x08\x0b\x0c\x0e-\x1f]'
    '|[\ud800-\udfff]'
)


def _sanitize_xml_text(text: str) -> str:
    """Remove characters that are not valid in XML 1.0, then repair bare ``&``."""
    text = _INVALID_XML_CHARS_RE.sub('', text)
    # Attempt to fix bare ampersands that aren't part of a known entity.
    # We do this conservatively: if `&` is followed by `#` or `[a-zA-Z]+;`,
    # it's a valid entity reference. Otherwise we escape the `&`.
    text = re.sub(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
    return text


def _local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].lower()


def _child_text(elem: ET.Element, name: str) -> str:
    for child in list(elem):
        if _local_name(child.tag) == name:
            return (child.text or '').strip()
    return ''


def parse_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return value


def parse_feed(url: str) -> list[dict[str, Any]]:
    raw, _ = fetch_url(url)
    text = raw.decode('utf-8', 'replace')
    text = _sanitize_xml_text(text)
    root = ET.fromstring(text)
    channel = root.find('./channel')
    if channel is not None:
        entries = channel.findall('./item')
        return [parse_item(item) for item in entries]
    entries = [e for e in root if _local_name(e.tag) == 'entry']
    return [parse_atom_entry(e) for e in entries]


def parse_item(item: ET.Element) -> dict[str, Any]:
    return {
        'title': _child_text(item, 'title') or '',
        'link': _child_text(item, 'link') or '',
        'guid': _child_text(item, 'guid') or _child_text(item, 'id') or '',
        'summary_html': _child_text(item, 'description') or _child_text(item, 'encoded') or '',
        'published_at': parse_datetime(_child_text(item, 'pubdate') or _child_text(item, 'date')),
        'raw': {child.tag: child.text for child in list(item)},
    }


def parse_atom_entry(entry: ET.Element) -> dict[str, Any]:
    link = ''
    for child in list(entry):
        if _local_name(child.tag) == 'link':
            link = child.attrib.get('href', '') or child.text or ''
            break
    return {
        'title': _child_text(entry, 'title') or '',
        'link': link,
        'guid': _child_text(entry, 'id') or '',
        'summary_html': _child_text(entry, 'summary') or _child_text(entry, 'content') or '',
        'published_at': parse_datetime(_child_text(entry, 'updated') or _child_text(entry, 'published')),
        'raw': {child.tag: child.text for child in list(entry)},
    }


def extract_goodreads_book_refs(html_blob: str) -> list[dict[str, str]]:
    html_blob = html_blob or ''
    refs: list[dict[str, str]] = []
    for match in BOOK_URL_RE.finditer(html_blob):
        book_id = match.group(1)
        url = match.group(0)
        snippet = html_blob[max(0, match.start() - 300): match.end() + 300]
        title = ''
        alt = ALT_RE.search(snippet)
        if alt:
            title = html.unescape(alt.group(1)).strip()
        if not title:
            # Goodreads slugs usually look like title-with-hyphens; use last segment as fallback.
            slug = url.split('/book/show/', 1)[-1].split('?', 1)[0]
            slug = slug.split('-', 1)[-1] if '-' in slug else slug
            title = html.unescape(slug.replace('-', ' ')).strip().title()
        author = ''
        plain = html_to_text(snippet)
        am = AUTHOR_RE.search(plain)
        if am:
            author = am.group(1).strip()
        refs.append({'goodreads_id': book_id, 'url': url, 'title': title, 'author': author})
    if not refs:
        plain = html_to_text(html_blob)
        if plain:
            refs.append({'goodreads_id': '', 'url': '', 'title': plain[:200], 'author': ''})
    seen = set()
    out = []
    for ref in refs:
        key = (ref.get('goodreads_id') or '', ref.get('title') or '')
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def normalized_feed_items(url: str) -> list[dict[str, Any]]:
    items = parse_feed(url)
    out: list[dict[str, Any]] = []
    for item in items:
        desc = item.get('summary_html') or ''
        refs = extract_goodreads_book_refs(desc) if 'goodreads.com' in (item.get('link') or desc) else []
        if refs:
            for ref in refs:
                out.append({**item, **ref})
        else:
            out.append({**item, 'goodreads_id': '', 'author': '', 'url': item.get('link') or ''})
    return out

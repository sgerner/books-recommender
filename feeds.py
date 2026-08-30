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
AUTHOR_RE = re.compile(r"author:\s*(.+?)(?:\s{2,}|\s+name:|\s+rating:|$)", re.I | re.S)


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
            return ''.join(child.itertext()).strip()
    return ''


def _first_author_text(elem: ET.Element) -> str:
    """Read common RSS/Atom author fields, including nested Atom ``name``."""
    for child in list(elem):
        if _local_name(child.tag) in {'author_name', 'creator', 'author'}:
            value = ''.join(child.itertext()).strip()
            if value:
                return value
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
    title = _child_text(item, 'title') or ''
    # Collect all <category> texts (some feeds put author as first category)
    categories = []
    for child in list(item):
        if _local_name(child.tag) == 'category' and child.text:
            categories.append(child.text.strip())
    return {
        'title': title,
        'link': _child_text(item, 'link') or '',
        'guid': _child_text(item, 'guid') or _child_text(item, 'id') or '',
        'summary_html': _child_text(item, 'description') or _child_text(item, 'encoded') or '',
        'published_at': parse_datetime(_child_text(item, 'pubdate') or _child_text(item, 'date')),
        'categories': categories,
        'author': _first_author_text(item),
        'raw': {child.tag: ''.join(child.itertext()).strip() for child in list(item)},
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
        'author': _first_author_text(entry),
        'raw': {child.tag: ''.join(child.itertext()).strip() for child in list(entry)},
    }


def extract_goodreads_book_refs(html_blob: str) -> list[dict[str, str]]:
    html_blob = html_blob or ''
    refs: list[dict[str, str]] = []
    for match in BOOK_URL_RE.finditer(html_blob):
        book_id = match.group(1)
        url = match.group(0)
        snippet = html_blob[max(0, match.start() - 900): match.end() + 1200]
        title = ''
        alt = ALT_RE.search(snippet)
        if alt:
            title = html.unescape(alt.group(1)).strip()
        if not title:
            slug = url.split('/book/show/', 1)[-1].split('?', 1)[0]
            slug = slug.split('-', 1)[-1] if '-' in slug else slug
            title = html.unescape(slug.replace('-', ' ')).strip().title()

        author = ''
        inline = re.match(r'^(.+?)\s+by\s+(.+?)\s*$', title, re.I)
        if inline:
            possible_author = re.sub(r'\s+', ' ', html.unescape(inline.group(2))).strip(' ,:;')
            if 1 <= len(possible_author.split()) <= 6 and not re.search(r'(?i)\b(?:the|and|with|for|from|author)\b', possible_author):
                title = inline.group(1).strip(' -–—:')
                author = possible_author
        if not author:
            structured = re.search(
                r"""<(?:span|a)[^>]*(?:class=["'][^"']*\bauthorName\b[^"']*["']|itemprop=["']name["'])[^>]*>(.*?)</(?:span|a)>""",
                snippet,
                re.I | re.S,
            )
            if structured:
                author = re.sub(r'\s+', ' ', html_to_text(structured.group(1))).strip()
        if not author:
            plain = html_to_text(snippet)
            am = AUTHOR_RE.search(plain)
            if am:
                author = re.sub(r'\s+', ' ', am.group(1)).strip(' ,:;')
        refs.append({'goodreads_id': book_id, 'url': url, 'title': title, 'author': author})
    if not refs:
        plain = html_to_text(html_blob)
        if plain:
            refs.append({'goodreads_id': '', 'url': '', 'title': plain[:200], 'author': ''})
    seen = set()
    out = []
    for ref in refs:
        key = ref.get('goodreads_id') or (ref.get('url') or ref.get('title') or '').lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _split_title_author(title: str) -> tuple[str, str]:
    """Split 'Title - Author' or 'Title: Subtitle - Author' patterns.
    Returns (title, author). Only splits on ' - ' when the part after looks
    like a person name (starts with capital, no lowercase after space, etc.)."""
    if ' - ' not in title:
        return title, ''
    # Try splitting on last ' - ' first (handles subtitles with hyphens)
    parts = title.rsplit(' - ', 1)
    if len(parts) == 2:
        candidate_author = parts[1].strip()
        # Heuristic: author part should look like a name
        # - Starts with uppercase letter
        # - No more than 4 words (most authors are 1-3 words)
        # - Doesn't look like a subtitle (no "A Novel", "A Story", etc.)
        if (candidate_author
                and candidate_author[0].isupper()
                and len(candidate_author.split()) <= 4
                and not re.match(r'(?i)^(a|an|the)\s', candidate_author)):
            return parts[0].strip(), candidate_author
    return title, ''


def normalized_feed_items(url: str) -> list[dict[str, Any]]:
    items = parse_feed(url)
    out: list[dict[str, Any]] = []
    non_author_categories = {
        'audio-books', 'audiobooks', 'fiction', 'nonfiction', 'non-fiction',
        'mysteries & thrillers', 'sci-fi & fantasy', 'biographies & memoirs',
    }
    for item in items:
        desc = item.get('summary_html') or ''
        refs = extract_goodreads_book_refs(desc) if 'goodreads.com' in (item.get('link') or desc) else []
        if refs:
            for ref in refs:
                out.append({**item, **ref})
        else:
            raw = item.get('raw') if isinstance(item.get('raw'), dict) else {}
            author = item.get('author') or ''
            if not author:
                for key, value in raw.items():
                    local = str(key).rsplit('}', 1)[-1].lower().replace(':', '_')
                    if local in {'author_name', 'creator', 'dc_creator', 'itunes_author', 'author'} and value:
                        author = str(value).strip()
                        break
            if ' name: ' in author:
                author = author.split(' name: ', 1)[0].strip()

            categories = item.get('categories') or []
            if not author:
                author = next((c for c in categories if c.strip().lower() not in non_author_categories), '')

            title = item.get('title', '')
            if not author and ' - ' in title:
                split_title, split_author = _split_title_author(title)
                if split_author:
                    title = split_title
                    author = split_author

            clean_desc = html_to_text(desc).strip()
            clean_title = (
                clean_desc
                if clean_desc and len(clean_desc) < len(title)
                and (' - ' in title or clean_desc.casefold() in title.casefold())
                else title
            )
            out.append({
                **item,
                'title': clean_title,
                'author': author,
                'goodreads_id': '',
                'url': item.get('link') or '',
            })
    return out

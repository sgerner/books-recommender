from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .feeds import fetch_url, normalized_feed_items
from .text import html_to_text, slugify

_JSON_LD_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
_AMP_RE = re.compile(r'&amp;')
_BOOKISH_PATH_RE = re.compile(r'/(book|books|title|titles|novel|novels|story|stories|series|read|reads|dp|gp/product)/', re.I)
_GOODREADS_BOOK_RE = re.compile(r'goodreads\.com/book/show/(\d+)', re.I)
_AUTHOR_RE = re.compile(r'\bby\s+([^|\n\r<]{2,80})', re.I)
_PLACEHOLDER_TITLE_RE = re.compile(r'^(?:saving|loading|more|here|read more|learn more|see all|view all)\s*[.…\.]*$', re.I)
_NON_BOOK_TITLE_RE = re.compile(r'^(?:see all of this year|readers[’\'] favorite|best books|new books recommended by readers|meet the winners|nominees here)', re.I)

KNOWN_BOOK_DOMAINS = (
    'goodreads.com',
    'bookshop.org',
    'amazon.com',
    'www.amazon.com',
    'books.apple.com',
    'barnesandnoble.com',
    'penguinrandomhouse.com',
    'harpercollins.com',
    'macmillan.com',
    'simonandschuster.com',
    'hachettebookgroup.com',
    'orbitbooks.net',
    'tor.com',
    'reactormag.com',
)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ''
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str]] = []
        self._current_link: dict[str, str] | None = None
        self._current_tag: str | None = None
        self._in_title = False
        self._script_type: str | None = None
        self._script_buf: list[str] | None = None
        self.scripts: list[dict[str, str]] = []
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs_map = {k.lower(): v for k, v in attrs}
        self._current_tag = tag.lower()
        if tag.lower() == 'title':
            self._in_title = True
        if tag.lower() == 'meta':
            name = attrs_map.get('property') or attrs_map.get('name') or attrs_map.get('itemprop')
            content = attrs_map.get('content') or ''
            if name and content:
                self.meta[name.lower()] = content
        if tag.lower() == 'a':
            self._current_link = {
                'href': attrs_map.get('href', '') or '',
                'text': '',
            }
        if tag.lower() == 'script':
            self._script_type = attrs_map.get('type', '') or ''
            self._script_buf = []

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == 'title':
            self._in_title = False
        if tag == 'a' and self._current_link is not None:
            self.links.append(self._current_link)
            self._current_link = None
        if tag == 'script' and self._script_buf is not None:
            self.scripts.append({'type': self._script_type or '', 'text': ''.join(self._script_buf)})
            self._script_buf = None
            self._script_type = None

    def handle_data(self, data: str):
        if self._in_title:
            self.title += data
        if self._current_link is not None:
            self._current_link['text'] += data
        if self._script_buf is not None:
            self._script_buf.append(data)
        self._text_chunks.append(data)

    def text(self) -> str:
        return re.sub(r'\s+', ' ', unescape(''.join(self._text_chunks))).strip()


@dataclass
class SourceFetchResult:
    source: dict[str, Any]
    items: list[dict[str, Any]]
    content_hash: str
    skipped: bool = False
    reason: str | None = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inject_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[key] = value
    return urlunparse(parsed._replace(query=urlencode(query)))


_NON_BOOK_PATH_PATTERNS = (
    'author/show/',
    'author/list/',
    'author/',
    '/authors/',
    '/a/',
    '/contributor/',
    '/writer/',
    '/profile/',
    '/user/show/',
)

def _bookish_link(href: str) -> bool:
    if not href:
        return False
    parsed = urlparse(href)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    # Reject author/profile/contributor pages which link to people, not books.
    if any(p in path for p in _NON_BOOK_PATH_PATTERNS):
        return False
    # Goodreads is noisy: blog, genre, choice-award and list pages are about
    # books but are not themselves book candidates. Only /book/show links are
    # safe to treat as candidate books.
    if 'goodreads.com' in host:
        return '/book/show/' in path
    return any(domain in host for domain in KNOWN_BOOK_DOMAINS) or bool(_BOOKISH_PATH_RE.search(path))


def _parse_json_ld_blocks(scripts: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for script in scripts:
        if 'ld+json' not in (script.get('type') or '').lower():
            continue
        raw = script.get('text') or ''
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.append(item)
        elif isinstance(obj, dict):
            out.append(obj)
    return out


def _normalize_candidate(candidate: dict[str, Any], source: dict[str, Any], source_uid: str) -> dict[str, Any]:
    candidate = dict(candidate)
    candidate.setdefault('source', source.get('name', 'source'))
    candidate.setdefault('source_uid', source_uid)
    candidate.setdefault('media_type', source.get('media_type', 'audiobook'))
    candidate.setdefault('url', '')
    candidate.setdefault('cover_url', '')
    candidate.setdefault('description', '')
    candidate.setdefault('published_at', None)
    candidate.setdefault('raw', {})
    return candidate


def _candidate_from_anchor(link: dict[str, str], page_text: str, html_blob: str, source: dict[str, Any], index: int) -> dict[str, Any] | None:
    href = (link.get('href') or '').strip()
    text = re.sub(r'\s+', ' ', unescape(link.get('text') or '')).strip()
    if not href or len(text) < 2:
        return None
    # Skip self-links / article headings that point back to the page itself.
    src_url = source.get('url') or ''
    if href == src_url or href.rstrip('/') == src_url.rstrip('/'):
        return None
    low_text = text.lower()
    if low_text.startswith(('writing in ', 'read in ', 'watch in ', 'listen to ', 'review round-up')):
        return None
    if _PLACEHOLDER_TITLE_RE.match(text) or _NON_BOOK_TITLE_RE.match(text):
        return None
    if not _bookish_link(href):
        return None
    candidate = {
        'title': text,
        'author': '',
        'url': href,
        'description': '',
        'cover_url': '',
        'raw': {'anchor_text': text, 'href': href, 'strategy': 'anchor'},
        'source_meta': {'strategy': 'anchor'},
    }
    # Look only near this specific href occurrence for a nearby "by Author" hint.
    href_idx = html_blob.lower().find(href.lower())
    if href_idx >= 0:
        snippet = html_blob[max(0, href_idx - 220): href_idx + 420]
        snippet_text = html_to_text(snippet)
        # Author extraction: match name parts (full words, initials, or dotted sequences like N.K.).
        # Separated by spaces or hyphens (for compound names like Smith-Jones).
        # This prevents crossing sentence boundaries or HTML tag remnants.
        WORD = r'[A-Z][\w\u00C0-\u024F]+'
        INITIAL = r'[A-Z]\.'
        INITIAL_SEQ = r'[A-Z](?:\.[A-Z])+\.?'
        PART = f'(?:{INITIAL_SEQ}|{WORD}|{INITIAL})'
        m = re.search(r'\bby\s+(' + PART + r'(?:[\s-]+' + PART + r')*)', snippet_text)
        if m:
            author = m.group(1).strip()
            # Truncate at em-dash, en-dash, colon, open paren.
            author = re.split(r'\s*—\s*|\s*–\s*|\s*:\s*|\s*\(', author, maxsplit=1)[0].strip()
            author = re.sub(r'\s+', ' ', author)
            # Reject sentence fragments and invalid names.
            author_lower = author.lower()
            if any(w in author_lower for w in (' the ', ' and ', ' with ', ' for ', ' from ', ' in ', ' of ')):
                author = ''
            elif len(author) < 2 or not re.search(r'[A-Za-z]{2,}', author):
                author = ''
            elif re.search(r'[<>{}]', author) or '&amp;' in author or '&lt;' in author:
                author = ''
            if author:
                candidate['author'] = author
    return _normalize_candidate(candidate, source, f"anchor:{index}:{slugify(text) or 'book'}")


def _candidate_from_goodreads_match(match: re.Match[str], html_blob: str, source: dict[str, Any], index: int) -> dict[str, Any]:
    book_id = match.group(1)
    url = match.group(0)
    snippet = html_blob[max(0, match.start() - 600): match.end() + 200]
    title = ''
    author = ''

    # Try img alt attribute first (Goodreads genre page layout).
    title_match = re.search(r'alt\s*=\s*["\x27]([^"\x27]{3,200})["\x27]', snippet, re.I)
    if title_match:
        raw_title = unescape(title_match.group(1)).strip()
        raw_title = re.sub(r'\s+', ' ', raw_title)
        if (not raw_title.isdigit()
                and not _PLACEHOLDER_TITLE_RE.match(raw_title)
                and not _NON_BOOK_TITLE_RE.match(raw_title)
                and not re.search(r'[<>{}]|&amp;|&lt;|class\s*=|width\s*=|src\s*=', raw_title)):
            title = raw_title

    # Fallback: heading or bookTitle span near the link.
    if not title:
        text_match = re.search(
            r'(?:<h[1-6][^>]*>|<strong[^>]*>|<span\s+class\s*=\s*"[^"]*bookTitle[^"]*"[^>]*>|<span\s+class\s*=\s*"[^"]*title[^"]*"[^>]*>)\s*(.+?)\s*(?:</h[1-6]>|</strong>|</span>)',
            snippet, re.I | re.S
        )
        if text_match:
            title = unescape(html_to_text(text_match.group(1))).strip()
            title = re.sub(r'\s+', ' ', title)

    # Last resort: extract from URL slug.
    if not title:
        slug = url.split('/book/show/', 1)[-1].split('?', 1)[0]
        slug = slug.split('-', 1)[-1] if '-' in slug else slug
        title = unescape(slug.replace('-', ' ')).strip().title()

    # Reject if title is just a number, placeholder text, category/navigation
    # copy, or HTML garbage.
    if (title.isdigit() or not title
            or _PLACEHOLDER_TITLE_RE.match(title)
            or _NON_BOOK_TITLE_RE.match(title)
            or re.search(r'[<>{}]|&amp;|&lt;|class=|width=|src=|bookCover', title)):
        return None

    # Author extraction with artifact filtering.
    snippet_text = html_to_text(snippet)
    author_match = _AUTHOR_RE.search(snippet_text)
    if author_match:
        candidate_author = author_match.group(1).strip()
        if not re.search(r'[<>{}"\x27]|width|height|class=|src=|bookCover|\.jpg|\.png|charset', candidate_author) and len(candidate_author.split()) <= 5:
            author = candidate_author

    return _normalize_candidate(
        {
            'title': title,
            'author': author,
            'url': url,
            'description': '',
            'cover_url': '',
            'raw': {'goodreads_id': book_id, 'strategy': 'goodreads-link'},
            'source_meta': {'strategy': 'goodreads-link'},
        },
        source,
        f'goodreads:{book_id}:{index}',
    )


def _candidates_from_json_ld(blocks: list[dict[str, Any]], source: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    for obj in blocks:
        if isinstance(obj.get('@graph'), list):
            graph = [x for x in obj['@graph'] if isinstance(x, dict)]
        else:
            graph = [obj]
        for item in graph:
            if not isinstance(item, dict):
                continue
            item_type = item.get('@type')
            if isinstance(item_type, list):
                item_type = ','.join(str(x) for x in item_type)
            item_type = str(item_type or '').lower()
            # ItemList pages often contain book recommendations or best-book roundups.
            if 'itemlist' in item_type or isinstance(item.get('itemListElement'), list):
                for pos, member in enumerate(item.get('itemListElement') or [], 1):
                    if not isinstance(member, dict):
                        continue
                    book = member.get('item') if isinstance(member.get('item'), dict) else member
                    if not isinstance(book, dict):
                        continue
                    title = book.get('name') or book.get('headline') or book.get('title') or ''
                    author = ''
                    if isinstance(book.get('author'), dict):
                        author = book['author'].get('name') or ''
                    elif isinstance(book.get('author'), str):
                        author = book.get('author') or ''
                    url = book.get('url') or ''
                    desc = book.get('description') or item.get('description') or ''
                    out.append(_normalize_candidate({
                        'title': str(title).strip(),
                        'author': str(author).strip(),
                        'url': str(url).strip(),
                        'description': str(desc).strip(),
                        'cover_url': str(book.get('image') or book.get('thumbnailUrl') or '').strip(),
                        'raw': {'json_ld': book, 'strategy': 'jsonld-itemlist', 'position': pos},
                        'source_meta': {'strategy': 'jsonld-itemlist', 'position': pos},
                    }, source, f'jsonld:item:{idx}:{pos}:{slugify(str(title)) or "book"}'))
                idx += 1
            elif 'book' in item_type:
                title = item.get('name') or item.get('headline') or item.get('title') or ''
                author = ''
                if isinstance(item.get('author'), dict):
                    author = item['author'].get('name') or ''
                elif isinstance(item.get('author'), str):
                    author = item.get('author') or ''
                if title and author:
                    out.append(_normalize_candidate({
                        'title': str(title).strip(),
                        'author': str(author).strip(),
                        'url': str(item.get('url') or '').strip(),
                        'description': str(item.get('description') or '').strip(),
                        'cover_url': str(item.get('image') or item.get('thumbnailUrl') or '').strip(),
                        'raw': {'json_ld': item, 'strategy': 'jsonld-book'},
                        'source_meta': {'strategy': 'jsonld-book'},
                    }, source, f'jsonld:book:{idx}:{slugify(str(title)) or "book"}'))
                    idx += 1
    return out


def _candidates_from_anchor_scan(page: _PageParser, page_text: str, html_blob: str, source: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, link in enumerate(page.links, 1):
        cand = _candidate_from_anchor(link, page_text, html_blob, source, idx)
        if cand:
            out.append(cand)
    # Goodreads links from raw html are especially useful on article roundup pages.
    for idx, match in enumerate(_GOODREADS_BOOK_RE.finditer(html_blob), 1):
        cand = _candidate_from_goodreads_match(match, html_blob, source, idx)
        if cand:
            out.append(cand)
    # De-duplicate by url+title.
    seen = set()
    deduped: list[dict[str, Any]] = []
    for cand in out:
        key = (cand.get('url', ''), cand.get('title', '').lower(), cand.get('author', '').lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _fetch_html_snapshot(url: str, timeout: int = 30) -> tuple[str, str]:
    raw, ctype = fetch_url(url, timeout=timeout)
    try:
        text = raw.decode('utf-8', errors='replace')
    except Exception:
        text = raw.decode('latin-1', errors='replace')
    return text, _sha256(raw)


def _fetch_json_snapshot(url: str, timeout: int = 30) -> tuple[dict[str, Any], str, bytes]:
    raw, _ = fetch_url(url, timeout=timeout)
    return json.loads(raw.decode('utf-8', errors='replace')), _sha256(raw), raw


def discover_source_items(source: dict[str, Any], cfg: dict[str, Any]) -> SourceFetchResult:
    kind = (source.get('kind') or 'web').lower()
    url = source.get('url') or ''
    if not url:
        return SourceFetchResult(source=source, items=[], content_hash='', skipped=True, reason='missing url')

    try:
        if kind in {'goodreads-rss', 'rss', 'atom'} or url.lower().endswith(('.rss', '.xml')):
            raw, _ = fetch_url(url)
            digest = _sha256(raw)
            if digest == (source.get('last_hash') or ''):
                return SourceFetchResult(source=source, items=[], content_hash=digest, skipped=True, reason='unchanged feed')
            items = normalized_feed_items(url)
            out: list[dict[str, Any]] = []
            for idx, item in enumerate(items, 1):
                uid = item.get('guid') or item.get('link') or f"{source.get('name','source')}:{idx}"
                out.append({
                    'title': item.get('title', '') or source.get('name', ''),
                    'author': item.get('author', ''),
                    'url': item.get('link') or item.get('url') or '',
                    'cover_url': item.get('cover_url', ''),
                    'description': item.get('summary_html') or '',
                    'published_at': item.get('published_at'),
                    'media_type': source.get('media_type', 'audiobook'),
                    'raw': item,
                    'source_meta': {'strategy': 'feed', 'uid': uid},
                    'source_uid': uid,
                })
            return SourceFetchResult(source=source, items=out, content_hash=digest)

        if kind in {'nyt-api', 'nytimes-api'} or ('nytimes.com' in url.lower() and 'api.nytimes.com' in url.lower()):
            from .config import resolve_nyt_api_key

            key = resolve_nyt_api_key(cfg)
            if not key:
                return SourceFetchResult(source=source, items=[], content_hash='', skipped=True, reason='missing NYT API key')
            fetch_url_final = url if 'api-key=' in url else _inject_query_param(url, 'api-key', key)
            data, digest, raw = _fetch_json_snapshot(fetch_url_final)
            if digest == (source.get('last_hash') or ''):
                return SourceFetchResult(source=source, items=[], content_hash=digest, skipped=True, reason='unchanged NYT data')
            results = data.get('results', {}) if isinstance(data, dict) else {}
            lists = results.get('lists', []) if isinstance(results, dict) else []
            out: list[dict[str, Any]] = []
            for list_idx, lst in enumerate(lists, 1):
                list_name = lst.get('display_name') or lst.get('list_name') or lst.get('name') or f'NYT List {list_idx}'
                list_slug = slugify(str(lst.get('list_name_encoded') or list_name or list_idx)) or f'list-{list_idx}'
                for rank, book in enumerate(lst.get('books', []) or [], 1):
                    title = book.get('title') or book.get('book_title') or ''
                    author = book.get('author') or book.get('book_author') or ''
                    uid = book.get('primary_isbn13') or book.get('primary_isbn10') or f'{list_slug}:{rank}:{slugify(title) or "book"}'
                    out.append({
                        'title': title,
                        'author': author,
                        'url': book.get('amazon_product_url') or book.get('book_uri') or book.get('book_review_link') or '',
                        'cover_url': book.get('book_image') or '',
                        'description': book.get('description') or '',
                        'published_at': results.get('published_date') or results.get('bestsellers_date') or '',
                        'media_type': source.get('media_type', 'audiobook'),
                        'raw': {'nyt': book, 'list': lst, 'all_results': results},
                        'source_meta': {'strategy': 'nyt-api', 'list_name': list_name, 'rank': rank},
                        'source_uid': f'nyt:{list_slug}:{uid}',
                    })
            return SourceFetchResult(source=source, items=out, content_hash=digest)

        if kind in {'openlibrary', 'open-library'} or 'openlibrary.org' in url.lower():
            data, digest, raw = _fetch_json_snapshot(url)
            if digest == (source.get('last_hash') or ''):
                return SourceFetchResult(source=source, items=[], content_hash=digest, skipped=True, reason='unchanged OpenLibrary data')
            works = data.get('works', []) if isinstance(data, dict) else []
            out: list[dict[str, Any]] = []
            for idx, work in enumerate(works, 1):
                title = work.get('title') or ''
                author_list = work.get('author_name') or []
                author = author_list[0] if author_list else ''
                work_key = work.get('key', '')
                work_url = f'https://openlibrary.org{work_key}' if work_key else ''
                cover_id = work.get('cover_i')
                cover_url = f'https://covers.openlibrary.org/b/id/{cover_id}-M.jpg' if cover_id else ''
                year = work.get('first_publish_year')
                published_at = f'{year}-01-01T00:00:00Z' if year else ''
                series_list = work.get('series_name') or []
                series = series_list[0] if series_list else ''
                edition_key = work.get('cover_edition_key') or ''
                out.append({
                    'title': title,
                    'author': author,
                    'url': work_url,
                    'cover_url': cover_url,
                    'description': '',
                    'published_at': published_at,
                    'media_type': source.get('media_type', 'audiobook'),
                    'raw': {'openlibrary': work},
                    'source_meta': {'strategy': 'openlibrary', 'rank': idx, 'edition_key': edition_key, 'series': series},
                    'source_uid': f'ol:{work_key}',
                })
            return SourceFetchResult(source=source, items=out, content_hash=digest)

        # Generic web article/page handling.
        html_blob, digest = _fetch_html_snapshot(url)
        if digest == (source.get('last_hash') or ''):
            return SourceFetchResult(source=source, items=[], content_hash=digest, skipped=True, reason='unchanged web page')
        parser = _PageParser()
        parser.feed(html_blob)
        page_text = parser.text()
        json_ld_objects = _parse_json_ld_blocks(parser.scripts)
        items = _candidates_from_json_ld(json_ld_objects, source)
        items.extend(_candidates_from_anchor_scan(parser, page_text, html_blob, source))

        # Plain-text fallback: look for Goodreads links or book-like title/author phrases in the visible text.
        if not items:
            text = html_to_text(html_blob)
            for idx, line in enumerate(re.split(r'\s{2,}|\n+', text), 1):
                line = line.strip()
                if len(line) < 8:
                    continue
                # More restrictive: title must be at least 2 words, author must be capitalized.
                m = re.match(r'^(.{4,120}?)\s+(?:by|—|–|-|:)\s+([A-Z][a-zA-Z\u00C0-\u024F][\w\u00C0-\u024F\s]{1,60})$', line, re.I)
                if not m:
                    continue
                title = m.group(1).strip(' -–—:')
                author = m.group(2).strip(' -–—:')
                # Reject titles that are too short or look like fragments.
                if len(title.split()) < 2 or len(title) < 4:
                    continue
                # Reject authors that look like sentence fragments.
                author_lower = author.lower()
                if any(w in author_lower for w in (' the ', ' and ', ' with ', ' for ', ' from ')):
                    continue
                # Clean up trailing punctuation from author.
                author = re.sub(r'[.,;!?]+$', '', author).strip()
                if len(author) < 2 or not re.search(r'[A-Za-z]{2,}', author):
                    continue
                items.append(_normalize_candidate({
                    'title': title,
                    'author': author,
                    'url': url,
                    'description': line,
                    'cover_url': '',
                    'raw': {'strategy': 'plain-text-fallback', 'line': line},
                    'source_meta': {'strategy': 'plain-text-fallback', 'line': idx},
                }, source, f'plain:{idx}:{slugify(title) or "book"}'))

        # Final dedupe.
        seen = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            key = (
                (item.get('source_uid') or '').lower(),
                (item.get('title') or '').strip().lower(),
                (item.get('author') or '').strip().lower(),
                (item.get('url') or '').strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return SourceFetchResult(source=source, items=deduped, content_hash=digest)
    except Exception as e:
        return SourceFetchResult(source=source, items=[], content_hash=source.get('last_hash') or '', skipped=True, reason=str(e))

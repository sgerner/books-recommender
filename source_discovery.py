from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

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

def _canonicalize_url(href: str, base_url: str = '') -> str:
    """Resolve a source link to a stable absolute URL.

    Goodreads pages contain a mixture of absolute, protocol-relative, and
    root-relative links. Treat the hostname explicitly so a bare match from a
    regular expression does not get joined underneath the source path.
    """
    href = unescape((href or '').strip())
    if not href:
        return ''
    if href.startswith('//'):
        return f'https:{href}'
    if re.match(r'^(?:www\.)?goodreads\.com/', href, re.I):
        return f'https://www.goodreads.com/{href.split("/", 1)[1]}'
    return urljoin(base_url or '', href)


def _goodreads_id_from_url(url: str) -> str:
    match = re.search(r'(?:^|/)book/show/(\d+)(?:[-/?#]|$)', url or '', re.I)
    return match.group(1) if match else ''


def _decode_embedded_html(fragment: str) -> str:
    """Decode the JSON-escaped HTML Goodreads embeds in page state."""
    return (
        (fragment or '')
        .replace('\\/', '/')
        .replace('\\"', '"')
        .replace('\\n', '\n')
        .replace('\\r', '\r')
        .replace('\\t', '\t')
        .replace('\\u0026', '&')
    )


def _clean_author(value: Any) -> str:
    """Return a conservative person-name value, or an empty string."""
    text = re.sub(r'\s+', ' ', html_to_text(unescape(str(value or '')))).strip(' \t\r\n|:,-')
    text = re.sub(r'^by\s+', '', text, flags=re.I).strip()
    if text.casefold() == 'by':
        return ''
    if not text or len(text) > 120 or re.search(r'[<>{}]|&(?:amp|lt|gt);', text, re.I):
        return ''
    lower = f' {text.lower()} '
    if any(f' {word} ' in lower for word in ('the', 'and', 'with', 'for', 'from', 'in', 'of', 'author')):
        return ''
    if len(text.split()) > 6 or not re.search(r'[A-Za-z\u00C0-\u024F]{2,}', text):
        return ''
    return text


def _split_inline_title_author(text: str) -> tuple[str, str]:
    """Split a visible ``Title by Author`` label when the suffix is a name."""
    text = re.sub(r'\s+', ' ', unescape(text or '')).strip()
    match = re.match(r'^(.+?)\s+by\s+(.+?)\s*$', text, re.I)
    if not match:
        return text, ''
    author = _clean_author(match.group(2))
    if not author:
        return text, ''
    title = match.group(1).strip(' \t\r\n-–—:')
    return (title or text), author


def _structured_author_near(html_blob: str, position: int) -> str:
    """Extract the author from the book card surrounding an HTML link.

    Goodreads currently renders authors in ``authorName`` / ``bookAuthors``
    elements and commonly marks the name with ``itemprop=name``. Matching the
    card markup is more reliable than assuming the text is within N characters
    of the book link.
    """
    if position < 0:
        return ''
    start = max(0, position - 1200)
    end = min(len(html_blob), position + 2200)
    window = _decode_embedded_html(html_blob[start:end])
    patterns = (
        r"""<[^>]*class=["'][^"']*\bauthorName\b[^"']*["'][^>]*>(.*?)</[^>]+>""",
        r"""<[^>]*id=["'][^"']*bookAuthors[^"']*["'][^>]*>(.*?)</[^>]+>""",
        r"""<[^>]*itemprop=["']author["'][^>]*>(.*?)</[^>]+>""",
        r"""<(?:span|a)[^>]*itemprop=["']name["'][^>]*>(.*?)</(?:span|a)>""",
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, window, re.I | re.S):
            author = _clean_author(match.group(1))
            if author:
                absolute = start + match.start()
                matches.append((abs(absolute - position), author))
    if not matches:
        return ''
    return min(matches, key=lambda item: item[0])[1]


def _bookish_link(href: str, base_url: str = '') -> bool:
    canonical = _canonicalize_url(href, base_url)
    if not canonical:
        return False
    parsed = urlparse(canonical)
    host = parsed.netloc.lower().split(':', 1)[0]
    path = parsed.path.lower()
    # Reject author/profile/contributor pages which link to people, not books.
    if any(p in path for p in _NON_BOOK_PATH_PATTERNS):
        return False
    # Goodreads is noisy: blog, genre, choice-award and list pages are about
    # books but are not themselves book candidates. Only a numeric
    # /book/show/<id> URL is safe to treat as a candidate book.
    if host == 'goodreads.com' or host.endswith('.goodreads.com'):
        return bool(re.match(r'^/book/show/\d+(?:[-/]|$)', path))
    return any(domain == host or host.endswith('.' + domain) for domain in KNOWN_BOOK_DOMAINS) or bool(_BOOKISH_PATH_RE.search(path))


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


def _structured_author(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get('name') or '').strip()
    if isinstance(value, list):
        names = []
        for entry in value:
            name = _structured_author(entry)
            if name and name not in names:
                names.append(name)
        return ', '.join(names)
    return str(value or '').strip()


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
    src_url = source.get('url') or ''
    canonical_href = _canonicalize_url(href, src_url)
    if not canonical_href:
        return None
    # Skip self-links / article headings that point back to the page itself.
    if canonical_href.rstrip('/') == _canonicalize_url(src_url).rstrip('/'):
        return None
    low_text = text.lower()
    if low_text.startswith(('writing in ', 'read in ', 'watch in ', 'listen to ', 'review round-up')):
        return None
    title, inline_author = _split_inline_title_author(text)
    if _PLACEHOLDER_TITLE_RE.match(title) or _NON_BOOK_TITLE_RE.match(title):
        return None
    if not _bookish_link(canonical_href):
        return None

    candidate = {
        'title': title,
        'author': inline_author,
        'url': canonical_href,
        'description': '',
        'cover_url': '',
        'raw': {'anchor_text': text, 'href': canonical_href, 'strategy': 'anchor'},
        'source_meta': {'strategy': 'anchor'},
    }
    # Look around this specific href for structured Goodreads card metadata.
    href_idx = html_blob.lower().find(href.lower())
    if href_idx < 0:
        href_idx = html_blob.lower().find(canonical_href.lower())
    if href_idx >= 0:
        if not candidate['author']:
            candidate['author'] = _structured_author_near(html_blob, href_idx)
        # Fallback for article prose: use a bounded, cleaned visible-text window.
        if not candidate['author']:
            snippet = _decode_embedded_html(html_blob[max(0, href_idx - 220): href_idx + 900])
            snippet_text = html_to_text(snippet)
            WORD = r'[A-Z][\w\u00C0-\u024F]+'
            INITIAL = r'[A-Z]\.'
            INITIAL_SEQ = r'[A-Z](?:\.[A-Z])+\.?'
            PART = f'(?:{INITIAL_SEQ}|{WORD}|{INITIAL})'
            match = re.search(r'\bby\s+(' + PART + r'(?:[\s-]+' + PART + r')*)', snippet_text)
            if match:
                candidate['author'] = _clean_author(match.group(1))
    if candidate['author'] and title.lower().endswith(f" by {candidate['author']}".lower()):
        candidate['title'] = title[: -(len(candidate['author']) + 4)].strip()
    candidate['title'] = re.sub(r'\s+', ' ', candidate['title']).strip()
    return _normalize_candidate(
        candidate,
        source,
        f"goodreads:{_goodreads_id_from_url(canonical_href)}" if _goodreads_id_from_url(canonical_href)
        else f"anchor:{index}:{slugify(canonical_href) or 'book'}",
    )


def _candidate_from_goodreads_match(match: re.Match[str], html_blob: str, source: dict[str, Any], index: int) -> dict[str, Any] | None:
    book_id = match.group(1)
    raw_url = match.group(0)
    url = _canonicalize_url(raw_url, source.get('url') or '')
    snippet = _decode_embedded_html(html_blob[max(0, match.start() - 1200): match.end() + 1600])
    title = ''
    author = ''

    # Try img alt attribute first (Goodreads genre page layout).
    title_match = re.search(r'alt\s*=\s*["\x27]([^"\x27]{3,200})["\x27]', snippet, re.I)
    if title_match:
        raw_title = re.sub(r'\s+', ' ', unescape(title_match.group(1))).strip()
        raw_title, inline_author = _split_inline_title_author(raw_title)
        if (not raw_title.isdigit()
                and not _PLACEHOLDER_TITLE_RE.match(raw_title)
                and not _NON_BOOK_TITLE_RE.match(raw_title)
                and not re.search(r'[<>{}]|&amp;|&lt;|class\s*=|width\s*=|src\s*=', raw_title)):
            title = raw_title
            author = inline_author

    # Fallback: heading or bookTitle span near the link.
    if not title:
        text_match = re.search(
            r'(?:<h[1-6][^>]*>|<strong[^>]*>|<span\s+class\s*=\s*["\x27][^"\x27]*bookTitle[^"\x27]*["\x27][^>]*>|<span\s+class\s*=\s*["\x27][^"\x27]*title[^"\x27]*["\x27][^>]*>)\s*(.+?)\s*(?:</h[1-6]>|</strong>|</span>)',
            snippet, re.I | re.S
        )
        if text_match:
            title, author = _split_inline_title_author(html_to_text(text_match.group(1)))
            title = re.sub(r'\s+', ' ', unescape(title)).strip()

    # Fallback for escaped/raw anchors whose href match does not include a
    # surrounding heading or image alt attribute.
    if not title:
        anchor_text_match = re.search(r'>\s*([^<>]{3,200})\s*</a>', snippet, re.I | re.S)
        if anchor_text_match:
            title, anchor_author = _split_inline_title_author(html_to_text(anchor_text_match.group(1)))
            if not author:
                author = anchor_author
            title = re.sub(r'\s+', ' ', title).strip()

    # Last resort: extract from URL slug.
    if not title:
        slug = url.split('/book/show/', 1)[-1].split('?', 1)[0]
        slug = slug.split('-', 1)[-1] if '-' in slug else slug
        title = unescape(slug.replace('-', ' ')).strip().title()

    # Structured author data is the primary source. It is deliberately searched
    # after the title because the surrounding card can contain several labels.
    if not author:
        author = _structured_author_near(html_blob, match.start())
    if not author:
        author = _clean_author(_AUTHOR_RE.search(html_to_text(snippet)).group(1)) if _AUTHOR_RE.search(html_to_text(snippet)) else ''

    if (title.isdigit() or not title
            or _PLACEHOLDER_TITLE_RE.match(title)
            or _NON_BOOK_TITLE_RE.match(title)
            or re.search(r'[<>{}]|&amp;|&lt;|class=|width=|src=|bookCover', title)):
        return None

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
        f'goodreads:{book_id}',
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
                    author = _structured_author(book.get('author'))
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
                author = _structured_author(item.get('author'))
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

    # Merge the anchor and raw-regex strategies. A Goodreads ID is the stable
    # identity; other links use canonical URL plus title. Prefer whichever row
    # has an author/description/cover, while retaining useful raw provenance.
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in out:
        goodreads_id = _goodreads_id_from_url(cand.get('url') or '') or (cand.get('raw') or {}).get('goodreads_id', '')
        if goodreads_id:
            key = ('goodreads', str(goodreads_id))
        else:
            key = ('url', f"{(cand.get('url') or '').lower()}|{(cand.get('title') or '').lower()}")
        previous = merged.get(key)
        if previous is None:
            merged[key] = cand
            continue
        if not previous.get('author') and cand.get('author'):
            previous['author'] = cand['author']
        if len(cand.get('title') or '') < len(previous.get('title') or '') and cand.get('title'):
            previous['title'] = cand['title']
        for field in ('description', 'cover_url'):
            if not previous.get(field) and cand.get(field):
                previous[field] = cand[field]
        previous_raw = previous.setdefault('raw', {})
        candidate_raw = cand.get('raw') or {}
        if isinstance(previous_raw, dict) and isinstance(candidate_raw, dict):
            for raw_key, raw_value in candidate_raw.items():
                previous_raw.setdefault(raw_key, raw_value)
    return list(merged.values())


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
            items = normalized_feed_items(url, raw=raw)
            out: list[dict[str, Any]] = []
            for idx, item in enumerate(items, 1):
                goodreads_id = str(item.get('goodreads_id') or '').strip()
                uid = (
                    f'goodreads:{goodreads_id}'
                    if goodreads_id
                    else item.get('guid') or item.get('link') or f"{source.get('name','source')}:{idx}"
                )
                book_url = item.get('url') or item.get('link') or ''
                out.append({
                    'title': item.get('title', '') or source.get('name', ''),
                    'author': item.get('author', ''),
                    'url': book_url,
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

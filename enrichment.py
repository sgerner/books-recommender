"""Unified book metadata enrichment with provider fallback.

Primary: Goodreads scraping (rich metadata)
Fallback: OpenLibrary API (pages, ISBN, covers)
Both are free, no API key required.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Goodreads (scrape-based, no API key needed)
# ---------------------------------------------------------------------------

def _fetch_goodreads(goodreads_id: str, delay: float = 2.0) -> dict[str, Any] | None:
    """Fetch Goodreads book page and extract structured metadata."""
    if delay > 0:
        time.sleep(delay)

    url = f'https://www.goodreads.com/book/show/{goodreads_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None

    result: dict[str, Any] = {
        'pages': None,
        'series': None,
        'genres': [],
        'format': None,
        'language': None,
        'isbn13': None,
        'author': None,
    }

    # Try __NEXT_DATA__ (Apollo state) first — Goodreads current stack.
    next_data_match = re.search(
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            apollo = data.get('props', {}).get('pageProps', {}).get('apolloState', {})
            for key, value in apollo.items():
                if isinstance(value, dict) and value.get('__typename') == 'Book':
                    details = value.get('details', {})
                    if isinstance(details, dict):
                        result['pages'] = details.get('numPages')
                        result['format'] = details.get('format')
                        result['isbn13'] = details.get('isbn13')
                        lang = details.get('language', {})
                        if isinstance(lang, dict):
                            result['language'] = lang.get('name')

                    book_authors = value.get('authors') or value.get('bookAuthors') or value.get('authorsList') or []
                    if isinstance(book_authors, list):
                        for entry in book_authors:
                            if isinstance(entry, dict):
                                author_name = entry.get('name') or entry.get('authorName')
                                if not author_name and entry.get('__ref') in apollo:
                                    author_obj = apollo.get(entry['__ref'])
                                    if isinstance(author_obj, dict):
                                        author_name = author_obj.get('name') or author_obj.get('authorName')
                                if author_name:
                                    result['author'] = str(author_name).strip()
                                    break
                            elif isinstance(entry, str) and entry.strip():
                                result['author'] = entry.strip()
                                break
                    if not result['author']:
                        for key_name in ('author', 'authorName', 'primaryAuthor'):
                            value_name = value.get(key_name)
                            if isinstance(value_name, dict):
                                value_name = value_name.get('name')
                            if value_name:
                                result['author'] = str(value_name).strip()
                                break

                    book_genres = value.get('bookGenres', [])
                    if isinstance(book_genres, list):
                        genres = []
                        for g in book_genres:
                            if isinstance(g, dict):
                                genre = g.get('genre', {})
                                if isinstance(genre, dict) and genre.get('name'):
                                    genres.append(genre['name'])
                        result['genres'] = genres

                    book_series = value.get('bookSeries', [])
                    if isinstance(book_series, list) and book_series:
                        series_entry = book_series[0]
                        if isinstance(series_entry, dict):
                            position = series_entry.get('userPosition')
                            series_ref = series_entry.get('series', {})
                            if isinstance(series_ref, dict):
                                ref_key = series_ref.get('__ref', '')
                                if ref_key and ref_key in apollo:
                                    series_obj = apollo[ref_key]
                                    if isinstance(series_obj, dict):
                                        series_name = series_obj.get('title') or series_obj.get('name', '')
                                        if position:
                                            result['series'] = f'{series_name}, #{position}'
                                        elif series_name:
                                            result['series'] = series_name
                    break
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback/augmentation: JSON-LD often contains author even when Apollo
    # omits it. Merge only missing fields so structured Apollo data wins.
    json_ld_match = re.search(
        r"""<script[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>""",
        html, re.DOTALL | re.IGNORECASE,
    )
    if json_ld_match:
        try:
            ld_data = json.loads(json_ld_match.group(1))
            ld_items = ld_data if isinstance(ld_data, list) else [ld_data]
            for ld_item in ld_items:
                if not isinstance(ld_item, dict):
                    continue
                pages = ld_item.get('numberOfPages')
                if not result['pages'] and pages:
                    try:
                        result['pages'] = int(pages)
                    except (ValueError, TypeError):
                        pass
                if not result['format']:
                    result['format'] = ld_item.get('bookFormat')
                if not result['language']:
                    result['language'] = ld_item.get('inLanguage')
                if not result['isbn13']:
                    isbn_value = ld_item.get('isbn', '')
                    if isbn_value and len(str(isbn_value).replace('-', '')) == 13:
                        result['isbn13'] = str(isbn_value).replace('-', '')
                if not result['author']:
                    ld_author = ld_item.get('author')
                    if isinstance(ld_author, dict):
                        ld_author = ld_author.get('name')
                    elif isinstance(ld_author, list):
                        ld_author = next((a.get('name') for a in ld_author if isinstance(a, dict) and a.get('name')), None)
                    if ld_author:
                        result['author'] = str(ld_author).strip()
                if result['author']:
                    break
        except (json.JSONDecodeError, TypeError):
            pass

    # Check if we got anything useful — if all fields are empty, treat as failure.
    has_data = any(result[key] for key in ('pages', 'series', 'genres', 'isbn13', 'author', 'format', 'language'))
    return result if has_data else None


# ---------------------------------------------------------------------------
# OpenLibrary (REST API, no key needed)
# ---------------------------------------------------------------------------

def _normalize_title(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (value or '').casefold()).strip()


def _title_variants(title: str) -> set[str]:
    value = re.sub(r'\s+', ' ', (title or '').strip())
    variants = {_normalize_title(value)} if value else set()
    base = re.sub(r'\s*\([^)]*(?:#|book|vol\.?|volume|part|edition|unabridged)[^)]*\)\s*$', '', value, flags=re.I).strip()
    if base:
        variants.add(_normalize_title(base))
    without_unabridged = re.sub(r'\s*\(?(?:unabridged|audiobook)\)?\s*$', '', value, flags=re.I).strip()
    if without_unabridged:
        variants.add(_normalize_title(without_unabridged))
    return {v for v in variants if v}


def _doc_authors(doc: dict[str, Any]) -> list[str]:
    values = doc.get('author_name') or doc.get('authors') or []
    if isinstance(values, str):
        values = [values]
    return [str(name).strip() for name in values if str(name).strip()]


def _author_matches(doc: dict[str, Any], author: str) -> bool:
    target = _normalize_title(author)
    if not target:
        return False
    return any(target == _normalize_title(name) or target in _normalize_title(name) or _normalize_title(name) in target for name in _doc_authors(doc))


def _fetch_openlibrary_title(title: str, author: str = '', delay: float = 1.0) -> dict[str, Any] | None:
    """Search OpenLibrary and accept only an exact title match.

    ``limit=1`` is unsafe when an author is missing because Open Library's
    ranking may choose a translation, adaptation, or unrelated near-match.
    """
    if delay > 0:
        time.sleep(delay)

    search_titles = [title]
    base_title = re.sub(
        r'\s*\([^)]*(?:#|book|vol\.?|volume|part|edition|unabridged)[^)]*\)\s*$',
        '',
        title,
        flags=re.I,
    ).strip()
    if base_title and base_title != title:
        search_titles.append(base_title)

    headers = {
        'User-Agent': 'BooksRecommender/1.0 (enrichment)',
        'Accept': 'application/json',
    }
    docs: list[dict[str, Any]] = []
    variants: set[str] = set()
    for search_title in search_titles:
        params = f'title={urllib.parse.quote(search_title)}&limit=20'
        url = f'https://openlibrary.org/search.json?{params}'
        if author:
            url += f'&author={urllib.parse.quote(author)}'
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            continue
        variants = _title_variants(search_title) | _title_variants(title)
        docs = [doc for doc in (data.get('docs', []) if isinstance(data, dict) else []) if isinstance(doc, dict)]
        exact_docs = [doc for doc in docs if _normalize_title(doc.get('title', '')) in variants]
        if exact_docs:
            docs = exact_docs
            break
        docs = []
    if not docs:
        return None

    def rank(doc: dict[str, Any]) -> tuple:
        languages = {str(lang).lower() for lang in (doc.get('language') or [])}
        try:
            first_year = int(doc.get('first_publish_year') or 0)
        except (TypeError, ValueError):
            first_year = 0
        return (
            0 if author and _author_matches(doc, author) else 1,
            0 if _doc_authors(doc) else 1,
            0 if 'eng' in languages else 1,
            -first_year,
            -(doc.get('edition_count') or 0),
        )

    doc = min(exact_docs, key=rank)
    pages = doc.get('number_of_pages_median') or doc.get('number_of_pages')
    result: dict[str, Any] = {
        'pages': pages,
        'series': None,
        'genres': list(doc.get('subject_facet', [])[:5]) if doc.get('subject_facet') else [],
        'format': None,
        'language': (doc.get('language') or [None])[0],
        'isbn13': None,
        'author': (_doc_authors(doc) or [None])[0],
        'first_published': doc.get('first_publish_year'),
        'cover_url': f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg" if doc.get('cover_i') else None,
    }

    for isbn in doc.get('isbn', []) or []:
        if isinstance(isbn, str) and len(isbn.replace('-', '')) == 13:
            result['isbn13'] = isbn.replace('-', '')
            break
    return result


def _fetch_openlibrary_isbn(isbn: str, delay: float = 1.0) -> dict[str, Any] | None:
    """Fetch metadata from OpenLibrary by ISBN."""
    if delay > 0:
        time.sleep(delay)

    isbn_clean = isbn.replace('-', '')
    url = f'https://openlibrary.org/isbn/{isbn_clean}.json'
    headers = {
        'User-Agent': 'BooksRecommender/1.0 (enrichment)',
        'Accept': 'application/json',
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    pages = data.get('number_of_pages')
    authors = data.get('authors') or []
    author = next((entry.get('name') for entry in authors if isinstance(entry, dict) and entry.get('name')), None)
    if not author:
        statement = str(data.get('by_statement') or '').strip()
        author_match = re.match(r'by\s+(.+?)(?:\s*;|\s*\(|$)', statement, re.I)
        author = author_match.group(1).strip() if author_match else None
    languages = data.get('languages') or {}
    result: dict[str, Any] = {
        'pages': pages,
        'series': None,
        'genres': list((data.get('subjects') or [])[:5]),
        'format': None,
        'language': next(iter(languages), None) if isinstance(languages, dict) else None,
        'isbn13': isbn_clean if len(isbn_clean) == 13 else None,
        'author': author,
    }
    return result


# ---------------------------------------------------------------------------
# Public API: unified enrichment with fallback
# ---------------------------------------------------------------------------

def enrich_book_metadata(
    title: str,
    author: str = '',
    goodreads_id: str | None = None,
    isbn: str | None = None,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Enrich book metadata by merging trustworthy partial results.

    Chain:
        1. Goodreads by ID, when available.
        2. OpenLibrary by ISBN when an ISBN is known or discovered.
        3. OpenLibrary by exact title, with optional author matching.

    A provider response is not considered complete merely because it contains
    pages or genres. This is important for repairing candidates whose author is
    missing. The returned ``provider`` records every provider that contributed
    data, in order.
    """
    result: dict[str, Any] = {
        'pages': None,
        'series': None,
        'genres': [],
        'format': None,
        'language': None,
        'isbn13': None,
        'author': None,
        'first_published': None,
        'cover_url': None,
        'provider': None,
        'fallback_used': False,
        'author_provider': None,
        'providers': [],
    }
    providers: list[str] = []

    def apply(metadata: dict[str, Any] | None, provider: str) -> None:
        if not metadata:
            return
        providers.append(provider)
        for key, value in metadata.items():
            if key in {'provider', 'fallback_used', 'providers'} or value is None or value == []:
                continue
            if key == 'author' and result['author']:
                continue
            if result.get(key) in (None, '', []):
                result[key] = value
                if key == 'author':
                    result['author_provider'] = provider

    gr = _fetch_goodreads(goodreads_id, delay=delay) if goodreads_id else None
    apply(gr, 'goodreads')

    # If the source did not provide an author, keep walking the chain even when
    # Goodreads supplied pages/genres. Prefer an ISBN discovered by Goodreads.
    source_author = (author or '').strip()
    isbn_for_fallback = isbn or result.get('isbn13')
    if isbn_for_fallback and (not goodreads_id or (not source_author and not result.get('author'))):
        ol_isbn = _fetch_openlibrary_isbn(str(isbn_for_fallback), delay=delay)
        apply(ol_isbn, 'openlibrary-isbn')

    # Title search is used when the author remains unresolved, or when every
    # stronger provider failed. The exact-match guard prevents near-title drift.
    if title and (not providers or (not source_author and not result.get('author'))):
        ol_title = _fetch_openlibrary_title(title, source_author, delay=delay)
        apply(ol_title, 'openlibrary-search')

    result['providers'] = providers
    result['provider'] = '+'.join(providers) if providers else 'none'
    result['fallback_used'] = bool(
        (goodreads_id and any(provider != 'goodreads' for provider in providers))
        or 'openlibrary-search' in providers
    )
    return result

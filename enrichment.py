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

    # Fallback: JSON-LD
    if not result['pages']:
        json_ld_match = re.search(
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE,
        )
        if json_ld_match:
            try:
                ld_data = json.loads(json_ld_match.group(1))
                if isinstance(ld_data, dict):
                    pages = ld_data.get('numberOfPages')
                    if pages:
                        try:
                            result['pages'] = int(pages)
                        except (ValueError, TypeError):
                            pass
                    if not result['format']:
                        result['format'] = ld_data.get('bookFormat')
                    if not result['language']:
                        result['language'] = ld_data.get('inLanguage')
                    if not result['isbn13']:
                        isbn = ld_data.get('isbn', '')
                        if isbn and len(str(isbn).replace('-', '')) == 13:
                            result['isbn13'] = str(isbn).replace('-', '')
            except json.JSONDecodeError:
                pass

    # Check if we got anything useful — if all None/empty, treat as failure.
    has_data = result['pages'] or result['series'] or result['genres'] or result['isbn13']
    return result if has_data else None


# ---------------------------------------------------------------------------
# OpenLibrary (REST API, no key needed)
# ---------------------------------------------------------------------------

def _fetch_openlibrary_title(title: str, author: str = '', delay: float = 1.0) -> dict[str, Any] | None:
    """Search OpenLibrary by title and return enriched metadata."""
    if delay > 0:
        time.sleep(delay)

    params = f'title={urllib.parse.quote(title)}&limit=1'
    url = f'https://openlibrary.org/search.json?{params}'
    if author:
        url += f'&author={urllib.parse.quote(author)}'

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

    docs = data.get('docs', [])
    if not docs:
        return None

    doc = docs[0]
    pages = doc.get('number_of_pages_median') or doc.get('number_of_pages')
    result: dict[str, Any] = {
        'pages': pages,
        'series': None,
        'genres': list(doc.get('subject_facet', [])[:5]) if doc.get('subject_facet') else [],
        'format': None,
        'language': None,
        'isbn13': None,
        'author': doc.get('author_name', [None])[0] if doc.get('author_name') else None,
        'first_published': doc.get('first_publish_year'),
        'cover_url': f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg" if doc.get('cover_i') else None,
    }

    # ISBN
    isbn13_list = doc.get('isbn', [])
    for isbn in isbn13_list:
        if isinstance(isbn, str) and len(isbn) == 13:
            result['isbn13'] = isbn
            break

    # Language
    lang_list = doc.get('language', [])
    if lang_list:
        result['language'] = lang_list[0]

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
    result: dict[str, Any] = {
        'pages': pages,
        'series': None,
        'genres': [],
        'format': None,
        'language': None,
        'isbn13': isbn_clean if len(isbn_clean) == 13 else None,
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
    """Enrich book metadata with provider fallback chain.

    Chain:
        1. Goodreads (if goodreads_id provided)
        2. OpenLibrary by ISBN (if isbn provided)
        3. OpenLibrary by title+author

    Returns dict with keys: pages, series, genres, format, language, isbn13,
    author (resolved), first_published, cover_url, provider, fallback_used.
    Missing values are None or empty lists.
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
    }

    # Try Goodreads first
    if goodreads_id:
        gr = _fetch_goodreads(goodreads_id, delay=delay)
        if gr:
            result.update({k: v for k, v in gr.items() if v is not None and v != []})
            result['provider'] = 'goodreads'
            return result

    # Fallback: OpenLibrary by ISBN
    if isbn:
        ol = _fetch_openlibrary_isbn(isbn, delay=delay)
        if ol:
            result.update({k: v for k, v in ol.items() if v is not None and v != []})
            result['provider'] = 'openlibrary-isbn'
            result['fallback_used'] = bool(goodreads_id)
            return result

    # Fallback: OpenLibrary by title+author
    if title:
        ol = _fetch_openlibrary_title(title, author, delay=delay)
        if ol:
            result.update({k: v for k, v in ol.items() if v is not None and v != []})
            result['provider'] = 'openlibrary-search'
            result['fallback_used'] = True
            return result

    result['provider'] = 'none'
    return result

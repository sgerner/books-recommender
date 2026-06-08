"""Goodreads book page scraper for enriching metadata."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any


def fetch_book_metadata(goodreads_id: str, delay: float = 2.0) -> dict[str, Any]:
    """Fetch and parse a Goodreads book page to extract enriched metadata.
    
    Args:
        goodreads_id: The Goodreads book ID
        delay: Seconds to wait before making the request (rate limiting)
    
    Returns:
        Dict with keys: pages, series, genres, format, language, isbn13
        Missing values are None or empty lists.
    """
    if delay > 0:
        time.sleep(delay)
    
    url = f'https://www.goodreads.com/book/show/{goodreads_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return {'error': str(e)}
    
    result: dict[str, Any] = {
        'pages': None,
        'series': None,
        'genres': [],
        'format': None,
        'language': None,
        'isbn13': None,
    }
    
    # Extract __NEXT_DATA__ JSON which contains structured Apollo state
    next_data_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            apollo = data.get('props', {}).get('pageProps', {}).get('apolloState', {})
            
            # Find the Book object
            for key, value in apollo.items():
                if isinstance(value, dict) and value.get('__typename') == 'Book':
                    # Extract from details
                    details = value.get('details', {})
                    if isinstance(details, dict):
                        result['pages'] = details.get('numPages')
                        result['format'] = details.get('format')
                        result['isbn13'] = details.get('isbn13')
                        
                        lang = details.get('language', {})
                        if isinstance(lang, dict):
                            result['language'] = lang.get('name')
                    
                    # Extract genres
                    book_genres = value.get('bookGenres', [])
                    if isinstance(book_genres, list):
                        genres = []
                        for g in book_genres:
                            if isinstance(g, dict):
                                genre = g.get('genre', {})
                                if isinstance(genre, dict) and genre.get('name'):
                                    genres.append(genre['name'])
                        result['genres'] = genres
                    
                    # Extract series
                    book_series = value.get('bookSeries', [])
                    if isinstance(book_series, list) and book_series:
                        # Take the first series
                        series_entry = book_series[0]
                        if isinstance(series_entry, dict):
                            # Get position from userPosition
                            position = series_entry.get('userPosition')
                            
                            # Get series reference
                            series_ref = series_entry.get('series', {})
                            if isinstance(series_ref, dict):
                                ref_key = series_ref.get('__ref', '')
                                # Resolve the reference from apollo state
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
    
    # Fallback: Extract JSON-LD data if __NEXT_DATA__ didn't work
    if not result['pages']:
        json_ld_match = re.search(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE)
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
                        if isbn and len(isbn.replace('-', '')) == 13:
                            result['isbn13'] = isbn.replace('-', '')
            except json.JSONDecodeError:
                pass
    
    return result

"""Tests for books_recommender: author extraction, enrichment, and scoring."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Author extraction tests (source_discovery.py)
# ---------------------------------------------------------------------------

class TestAuthorExtraction:
    """Test improved author extraction edge cases."""

    def _make_source(self, url='https://example.com/article'):
        return {
            'name': 'test-source',
            'url': url,
            'media_type': 'audiobook',
            'weight': 0.2,
        }

    def test_basic_author_from_anchor(self):
        """Standard 'by Author' after a book link."""
        from books_recommender.source_discovery import _candidate_from_anchor

        # Simulate a page where the href appears near "by John Scalzi"
        href = 'https://www.goodreads.com/book/show/12345'
        html_blob = f'<a href="{href}">The Last Emperox</a> by John Scalzi. Great book.'
        source = self._make_source()

        link = {'href': href, 'text': 'The Last Emperox'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        assert result is not None
        assert result['author'] == 'John Scalzi'
        assert result['title'] == 'The Last Emperox'

    def test_author_with_html_tags_around_it(self):
        """Author name should not cross HTML tag boundaries."""
        from books_recommender.source_discovery import _candidate_from_anchor

        href = 'https://www.goodreads.com/book/show/99999'
        # The "by" and author are split across tags — author should stop cleanly.
        html_blob = f'<a href="{href}">Tusk</a> <span>by</span> <em>Ali</em> <em>Sharp</em> is out now.'
        source = self._make_source()

        link = {'href': href, 'text': 'Tusk'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        assert result is not None
        # Should extract "Ali" (first capitalized word after "by"), not the full HTML mess.
        # The regex is conservative — it gets what it can from cleaned text.
        assert result['author'] != ''
        assert 'Ali' in result['author']

    def test_author_with_parenthetical(self):
        """Author should be truncated at parentheses."""
        from books_recommender.source_discovery import _candidate_from_anchor

        href = 'https://www.goodreads.com/book/show/55555'
        html_blob = f'<a href="{href}">Book Title</a> by Marina J. Lostetter (sequel to ...) out now.'
        source = self._make_source()

        link = {'href': href, 'text': 'Book Title'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        assert result is not None
        assert result['author'] == 'Marina J. Lostetter'

    def test_author_with_em_dash(self):
        """Author should be truncated at em-dashes."""
        from books_recommender.source_discovery import _candidate_from_anchor

        href = 'https://www.goodreads.com/book/show/77777'
        html_blob = f'<a href="{href}">New Novel</a> by N.K. Jemisin — the acclaimed author.'
        source = self._make_source()

        link = {'href': href, 'text': 'New Novel'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        assert result is not None
        assert result['author'] == 'N.K. Jemisin'

    def test_author_sentence_fragment_rejected(self):
        """Authors that are sentence fragments should be rejected."""
        from books_recommender.source_discovery import _candidate_from_anchor

        href = 'https://www.goodreads.com/book/show/88888'
        # "by the author of ..." should not become the author.
        html_blob = f'<a href="{href}">Book Title</a> by the author of The Handmaid\'s Tale.'
        source = self._make_source()

        link = {'href': href, 'text': 'Book Title'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        assert result is not None
        # "the author" contains "the" — should be rejected.
        assert result['author'] == '' or 'the author' not in result['author'].lower()

    def test_plain_text_fallback(self):
        """Plain-text fallback should extract title by Author."""
        from books_recommender.source_discovery import discover_source_items

        # Mock the fetch to return plain text with book mentions.
        html = '''<html><body>
        <p>Some random text here.</p>
        <p>The Left Hand of Darkness by Ursula K. Le Guin</p>
        <p>More text.</p>
        <p>Foundation by Isaac Asimov</p>
        </body></html>'''

        source = self._make_source(url='https://example.com/books')
        with patch('books_recommender.source_discovery.fetch_url', return_value=(html.encode(), 'text/html')):
            with patch('books_recommender.source_discovery._sha256', return_value='abc123'):
                result = discover_source_items(source, {})

        # Should find at least the two books.
        titles = [item.get('title') for item in result.items]
        assert any('Left Hand' in t for t in titles), f"Expected 'Left Hand' in titles, got: {titles}"

    def test_no_html_remnants_in_author(self):
        """Author should not contain HTML entities or tags."""
        from books_recommender.source_discovery import _candidate_from_anchor

        href = 'https://www.goodreads.com/book/show/11111'
        html_blob = f'<a href="{href}">Book</a> by John&amp;Smith &lt;author&gt; out now.'
        source = self._make_source()

        link = {'href': href, 'text': 'Book'}
        result = _candidate_from_anchor(link, '', html_blob, source, 1)

        if result:
            assert '&amp;' not in (result.get('author') or '')
            assert '&lt;' not in (result.get('author') or '')
            assert '<' not in (result.get('author') or '')
            assert '>' not in (result.get('author') or '')


# ---------------------------------------------------------------------------
# Enrichment fallback tests (enrichment.py)
# ---------------------------------------------------------------------------

class TestEnrichment:
    """Test unified enrichment with provider fallback."""

    def test_enrichment_returns_none_when_all_fail(self):
        """Should return 'none' provider when all sources fail."""
        from books_recommender.enrichment import enrich_book_metadata

        with patch('books_recommender.enrichment._fetch_goodreads', return_value=None):
            with patch('books_recommender.enrichment._fetch_openlibrary_title', return_value=None):
                with patch('books_recommender.enrichment._fetch_openlibrary_isbn', return_value=None):
                    result = enrich_book_metadata(title='Unknown Book', author='Nobody')

        assert result['provider'] == 'none'
        assert result['pages'] is None

    def test_enrichment_uses_goodreads_first(self):
        """Should try Goodreads first when goodreads_id is provided."""
        from books_recommender.enrichment import enrich_book_metadata

        goodreads_result = {
            'pages': 320,
            'series': 'Example, #1',
            'genres': ['Science Fiction'],
            'format': 'Paperback',
            'language': 'English',
            'isbn13': '9781234567890',
        }

        with patch('books_recommender.enrichment._fetch_goodreads', return_value=goodreads_result):
            with patch('books_recommender.enrichment._fetch_openlibrary_title') as ol_mock:
                result = enrich_book_metadata(
                    title='Test Book', author='Author',
                    goodreads_id='12345'
                )

        assert result['provider'] == 'goodreads'
        assert result['pages'] == 320
        assert result['series'] == 'Example, #1'
        ol_mock.assert_not_called()

    def test_enrichment_falls_back_to_openlibrary_search(self):
        """Should fall back to OpenLibrary when Goodreads fails."""
        from books_recommender.enrichment import enrich_book_metadata

        ol_result = {
            'pages': 280,
            'series': None,
            'genres': ['Fantasy'],
            'format': None,
            'language': 'English',
            'isbn13': '9780987654321',
            'author': 'Test Author',
            'first_published': 2023,
            'cover_url': 'https://covers.openlibrary.org/b/id/12345-M.jpg',
        }

        with patch('books_recommender.enrichment._fetch_goodreads', return_value=None):
            with patch('books_recommender.enrichment._fetch_openlibrary_isbn', return_value=None):
                with patch('books_recommender.enrichment._fetch_openlibrary_title', return_value=ol_result):
                    result = enrich_book_metadata(
                        title='Test Book', author='Author',
                        goodreads_id='12345'
                    )

        assert result['provider'] == 'openlibrary-search'
        assert result['fallback_used'] is True
        assert result['pages'] == 280

    def test_enrichment_falls_back_to_openlibrary_isbn(self):
        """Should try OpenLibrary ISBN before title search."""
        from books_recommender.enrichment import enrich_book_metadata

        ol_result = {
            'pages': 350,
            'series': None,
            'genres': [],
            'format': None,
            'language': None,
            'isbn13': '9781234567890',
        }

        with patch('books_recommender.enrichment._fetch_goodreads', return_value=None):
            with patch('books_recommender.enrichment._fetch_openlibrary_isbn', return_value=ol_result):
                with patch('books_recommender.enrichment._fetch_openlibrary_title') as title_mock:
                    result = enrich_book_metadata(
                        title='Test Book', author='Author',
                        isbn='9781234567890'
                    )

        assert result['provider'] == 'openlibrary-isbn'
        # fallback_used is True only when goodreads_id was provided but failed.
        # Here no goodreads_id was passed, so this is the primary path, not a fallback.
        assert result['fallback_used'] is False
        title_mock.assert_not_called()


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLIDigest:
    """Test digest command output and sanity checks."""

    def _make_db(self):
        """Create a temp SQLite DB with minimal schema."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            source TEXT, source_uid TEXT, title TEXT, author TEXT,
            rating INTEGER, goodreads_id TEXT, storygraph_id TEXT, isbn TEXT,
            note_path TEXT, tags_json TEXT, themes_json TEXT, summary TEXT,
            review TEXT, body TEXT, raw_json TEXT,
            created_at TEXT, updated_at TEXT,
            UNIQUE(source, source_uid)
        )''')
        conn.execute('''CREATE TABLE candidates (
            id INTEGER PRIMARY KEY,
            source TEXT, source_uid TEXT, title TEXT, author TEXT,
            url TEXT, cover_url TEXT, media_type TEXT, published_at TEXT,
            description TEXT, raw_json TEXT, score REAL, score_breakdown TEXT,
            status TEXT DEFAULT 'new', librarr_id TEXT, decided_at TEXT,
            created_at TEXT, updated_at TEXT,
            UNIQUE(source, source_uid)
        )''')
        conn.execute('''CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            entity_type TEXT, entity_id INTEGER, model TEXT,
            text_hash TEXT, dim INTEGER, vector_blob BLOB
        )''')
        conn.execute('''CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            kind TEXT, payload_json TEXT, created_at TEXT
        )''')
        conn.commit()
        return conn

    def test_digest_empty_candidates_shows_alert(self):
        """Digest with no candidates should show a warning, not crash."""
        from books_recommender.cli import cmd_digest
        import argparse

        conn = self._make_db()
        args = argparse.Namespace(config=None, limit=10, format='markdown', min_candidates=3)

        with patch('books_recommender.cli.load_config', return_value={
            'db_path': ':memory:',
            'embeddings': {'enabled': False},
            'enrichment': {'enabled': False},
            'source_weights': {},
            'recommendation': {},
        }):
            with patch('books_recommender.cli.connect', return_value=conn):
                with patch('books_recommender.cli.init_db'):
                    with patch('books_recommender.cli._score_pending', return_value=([], [])):
                        with patch('books_recommender.cli._candidate_match_keys', return_value=set()):
                            import io, sys
                            old_stdout = sys.stdout
                            sys.stdout = io.StringIO()
                            try:
                                cmd_digest(args)
                                output = sys.stdout.getvalue()
                            finally:
                                sys.stdout = old_stdout

        assert 'WARNING' in output
        assert '0 candidates' in output

    def test_digest_json_format(self):
        """Digest JSON output should be valid JSON."""
        from books_recommender.cli import cmd_digest
        import argparse

        conn = self._make_db()
        args = argparse.Namespace(config=None, limit=10, format='json', min_candidates=1)

        candidate = {
            'id': 1, 'title': 'Test Book', 'author': 'Author',
            'url': '', 'cover_url': '', 'media_type': 'audiobook',
            'published_at': None, 'description': '',
            'raw': {}, 'score': 85, 'status': 'new',
        }
        scored = {
            'score': 85, 'reasons': ['similar to liked books'],
            'similar_books': [], 'semantic_books': [],
        }

        with patch('books_recommender.cli.load_config', return_value={
            'db_path': ':memory:',
            'embeddings': {'enabled': False},
            'enrichment': {'enabled': False},
            'source_weights': {},
            'recommendation': {},
        }):
            with patch('books_recommender.cli.connect', return_value=conn):
                with patch('books_recommender.cli.init_db'):
                    with patch('books_recommender.cli._score_pending', return_value=([], [(candidate, scored)])):
                        with patch('books_recommender.cli._candidate_match_keys', return_value=set()):
                            import io, sys
                            old_stdout = sys.stdout
                            sys.stdout = io.StringIO()
                            try:
                                cmd_digest(args)
                                output = sys.stdout.getvalue()
                            finally:
                                sys.stdout = old_stdout

        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]['title'] == 'Test Book'


# ---------------------------------------------------------------------------
# Candidate is_banned tests
# ---------------------------------------------------------------------------

class TestCandidateBanned:
    """Test banned format detection (graphic novels)."""

    def test_graphic_novel_banned(self):
        from books_recommender.scoring import candidate_is_banned

        candidate = {
            'title': 'Saga Vol. 3',
            'author': 'Brian K. Vaughan',
            'description': 'A graphic novel about war.',
            'tags': [], 'themes': [],
        }
        assert candidate_is_banned(candidate) is True

    def test_regular_book_not_banned(self):
        from books_recommender.scoring import candidate_is_banned

        candidate = {
            'title': 'The Left Hand of Darkness',
            'author': 'Ursula K. Le Guin',
            'description': 'A science fiction novel about gender.',
            'tags': [], 'themes': [],
        }
        assert candidate_is_banned(candidate) is False

    def test_manga_banned(self):
        from books_recommender.scoring import candidate_is_banned

        candidate = {
            'title': 'Attack on Titan Vol. 1',
            'author': 'Hajime Isayama',
            'description': 'Manga about titans.',
            'tags': [], 'themes': [],
        }
        assert candidate_is_banned(candidate) is True

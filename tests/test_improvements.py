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
        conn.execute('''CREATE TABLE discord_messages (
            id INTEGER PRIMARY KEY,
            candidate_id INTEGER UNIQUE,
            channel_id TEXT, message_id TEXT UNIQUE, content TEXT,
            status TEXT DEFAULT 'posted', reaction TEXT, reaction_counts_json TEXT,
            librarr_id TEXT, posted_at TEXT, reacted_at TEXT, updated_at TEXT
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

    def test_enrich_candidates_accepts_sqlite_rows(self):
        """Regression test: _enrich_candidates should work with sqlite3.Row inputs."""
        from books_recommender.cli import _enrich_candidates

        conn = self._make_db()
        conn.execute(
            '''INSERT INTO candidates (
                source, source_uid, title, author, url, cover_url, media_type,
                published_at, description, raw_json, score, score_breakdown,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
            (
                'openlibrary-trending', 'ol:/works/OL1W', 'Sample Book', 'Sample Author',
                'https://openlibrary.org/works/OL1W', '', 'audiobook',
                '2026-01-01T00:00:00Z', '', json.dumps({'isbn': '9780000000000'}),
                None, None, 'new',
            ),
        )
        conn.commit()
        row = conn.execute('SELECT * FROM candidates WHERE source_uid=?', ('ol:/works/OL1W',)).fetchone()

        fake_meta = {
            'provider': 'openlibrary-search',
            'pages': 320,
            'series': 'Example Series #1',
            'genres': ['Fantasy'],
            'isbn13': '9780000000000',
            'language': 'English',
            'first_published': 2026,
            'cover_url': 'https://covers.openlibrary.org/b/id/12345-M.jpg',
            'author': 'Sample Author',
        }

        with patch('books_recommender.cli.enrich_book_metadata', return_value=fake_meta):
            stats = _enrich_candidates(conn, [row], {'enrichment': {'enabled': True, 'delay': 0, 'max_per_run': 5}})

        assert stats['enriched'] == 1
        updated = conn.execute('SELECT raw_json, cover_url FROM candidates WHERE source_uid=?', ('ol:/works/OL1W',)).fetchone()
        raw = json.loads(updated['raw_json'])
        assert raw['pages'] == 320
        assert raw['genres'] == ['Fantasy']
        assert raw['enrichment_provider'] == 'openlibrary-search'
        assert updated['cover_url'] == 'https://covers.openlibrary.org/b/id/12345-M.jpg'


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


class TestDiscordWorkflow:
    def _make_db(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE candidates (
            id INTEGER PRIMARY KEY,
            source TEXT, source_uid TEXT, title TEXT, author TEXT,
            url TEXT, cover_url TEXT, media_type TEXT, published_at TEXT,
            description TEXT, raw_json TEXT, score REAL, score_breakdown TEXT,
            status TEXT DEFAULT 'new', librarr_id TEXT, decided_at TEXT,
            created_at TEXT, updated_at TEXT,
            UNIQUE(source, source_uid)
        )''')
        conn.execute('''CREATE TABLE discord_messages (
            id INTEGER PRIMARY KEY,
            candidate_id INTEGER UNIQUE,
            channel_id TEXT, message_id TEXT UNIQUE, content TEXT,
            status TEXT DEFAULT 'posted', reaction TEXT, reaction_counts_json TEXT,
            librarr_id TEXT, posted_at TEXT, reacted_at TEXT, updated_at TEXT
        )''')
        conn.execute('''CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER, action TEXT, note TEXT, channel TEXT, created_at TEXT
        )''')
        conn.commit()
        return conn

    def test_reaction_decision_prefers_up(self):
        from books_recommender.discord_workflow import _reaction_decision, THUMBS_UP, THUMBS_DOWN

        decision, counts = _reaction_decision({
            'reactions': [
                {'emoji': {'name': THUMBS_UP}, 'count': 2},
                {'emoji': {'name': THUMBS_DOWN}, 'count': 1},
            ]
        })
        assert decision == 'approve'
        assert counts[THUMBS_UP] == 2

    def test_reaction_decision_prefers_down(self):
        from books_recommender.discord_workflow import _reaction_decision, THUMBS_UP, THUMBS_DOWN

        decision, counts = _reaction_decision({
            'reactions': [
                {'emoji': {'name': THUMBS_UP}, 'count': 1},
                {'emoji': {'name': THUMBS_DOWN}, 'count': 2},
            ]
        })
        assert decision == 'reject'
        assert counts[THUMBS_DOWN] == 2

    def test_apple_audiobook_title_matches_canonical_library_row(self):
        from books_recommender.db import candidate_match_keys, normalize_candidate_match_key

        book_key = normalize_candidate_match_key("Carl's Doomsday Scenario (Dungeon Crawler Carl, #2)", 'Matt Dinniman')
        apple_keys = candidate_match_keys(
            "Carl's Doomsday Scenario: Dungeon Crawler Carl, Book 2 (Unabridged)",
            'Matt Dinniman',
            'apple-top-audiobooks',
        )
        assert book_key in apple_keys

    def test_sync_reactions_marks_approved_and_posts_to_librarr(self):
        from books_recommender.discord_workflow import sync_reactions, THUMBS_UP
        from books_recommender.db import upsert_discord_message
        import argparse

        conn = self._make_db()
        conn.execute(
            '''INSERT INTO candidates (
                id, source, source_uid, title, author, url, cover_url, media_type,
                published_at, description, raw_json, score, score_breakdown,
                status, created_at, updated_at
            ) VALUES (1, 'openlibrary-trending', 'ol:/works/OL1W', 'Sample Book', 'Sample Author',
                      'https://openlibrary.org/works/OL1W', '', 'audiobook',
                      '2026-01-01T00:00:00Z', '', '{}', 88, '{}', 'discord_pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'''
        )
        upsert_discord_message(
            conn,
            candidate_id=1,
            channel_id='123',
            message_id='456',
            content='**Sample Book** — Sample Author',
            status='posted',
        )

        fake_cfg = {
            'discord': {'enabled': True},
            'librarr': {'url': 'http://librarr:5050', 'wishlist_media_type': 'audiobook'},
        }

        with patch('books_recommender.discord_workflow.resolve_discord_token', return_value='token'):
            with patch('books_recommender.discord_workflow.discord_fetch_message', return_value={
                'reactions': [
                    {'emoji': {'name': THUMBS_UP}, 'count': 2},
                    {'emoji': {'name': '👎'}, 'count': 1},
                ]
            }):
                with patch('books_recommender.discord_workflow.LibrarrClient') as client_mock:
                    client_mock.return_value.add_to_wishlist.return_value = {'id': 99}
                    with patch('books_recommender.discord_workflow.discord_edit_message'):
                        result = sync_reactions(conn, fake_cfg)

        assert result['count'] == 1
        row = conn.execute('SELECT status, librarr_id FROM candidates WHERE id=1').fetchone()
        assert row['status'] == 'approved'
        assert row['librarr_id'] == '99'
        dm = conn.execute('SELECT status, reaction, librarr_id FROM discord_messages WHERE candidate_id=1').fetchone()
        assert dm['status'] == 'approved'
        assert dm['reaction'] == THUMBS_UP
        assert dm['librarr_id'] == '99'

    def test_post_recommendations_marks_pending_and_posts(self):
        from books_recommender.discord_workflow import post_recommendations

        conn = self._make_db()
        conn.execute(
            '''INSERT INTO candidates (
                id, source, source_uid, title, author, url, cover_url, media_type,
                published_at, description, raw_json, score, score_breakdown,
                status, created_at, updated_at
            ) VALUES (1, 'openlibrary-trending', 'ol:/works/OL1W', 'Sample Book', 'Sample Author',
                      'https://openlibrary.org/works/OL1W', '', 'audiobook',
                      '2026-01-01T00:00:00Z', '', '{}', NULL, NULL, 'new', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'''
        )
        conn.commit()

        candidate = {
            'id': 1,
            'source': 'openlibrary-trending',
            'source_uid': 'ol:/works/OL1W',
            'title': 'Sample Book',
            'author': 'Sample Author',
            'url': 'https://openlibrary.org/works/OL1W',
            'cover_url': '',
            'media_type': 'audiobook',
            'published_at': '2026-01-01T00:00:00Z',
            'description': '',
            'raw': {},
            'score': 88,
            'score_breakdown': {},
            'status': 'new',
        }
        scored = {'score': 88, 'reasons': ['Fits the profile'], 'similar_books': []}
        fake_cfg = {
            'discord': {'enabled': True, 'channel_id': '#books'},
            'librarr': {'url': 'http://librarr:5050', 'wishlist_media_type': 'audiobook'},
        }

        with patch('books_recommender.discord_workflow.resolve_discord_token', return_value='token'):
            with patch('books_recommender.discord_workflow._resolve_discord_channel_id', return_value='123'):
                with patch('books_recommender.discord_workflow._build_digest_candidates', return_value=([(candidate, scored)], None, 1)):
                    with patch('books_recommender.discord_workflow.discord_post_message', return_value={'id': '456'}):
                        with patch('books_recommender.discord_workflow.discord_add_reaction'):
                            result = post_recommendations(conn, fake_cfg, limit=1)

        assert result['posted'][0]['message_id'] == '456'
        row = conn.execute('SELECT status FROM candidates WHERE id=1').fetchone()
        assert row['status'] == 'discord_pending'
        dm = conn.execute('SELECT status, message_id FROM discord_messages WHERE candidate_id=1').fetchone()
        assert dm['status'] == 'posted'
        assert dm['message_id'] == '456'


class TestIngestionQualityRegressions:
    def test_goodreads_structured_card_author_and_stable_uid(self):
        from books_recommender.source_discovery import _candidate_from_anchor

        source = {
            'name': 'goodreads-science-fiction',
            'url': 'https://www.goodreads.com/genres/science-fiction',
            'media_type': 'audiobook',
        }
        href = '/book/show/12345-the-book'
        html = f'''<div class="bookBox">
          <a href="{href}"><img alt="The Book" /></a>
          <div id="bookAuthors"><a class="authorName"><span itemprop="name">E.L. Wilk</span></a></div>
        </div>'''

        result = _candidate_from_anchor({'href': href, 'text': 'The Book'}, '', html, source, 1)

        assert result is not None
        assert result['title'] == 'The Book'
        assert result['author'] == 'E.L. Wilk'
        assert result['url'] == 'https://www.goodreads.com/book/show/12345-the-book'
        assert result['source_uid'] == 'goodreads:12345'

    def test_goodreads_embedded_html_card_author(self):
        import re
        from books_recommender.source_discovery import _GOODREADS_BOOK_RE, _candidate_from_goodreads_match

        source = {
            'name': 'goodreads-science-fiction',
            'url': 'https://www.goodreads.com/genres/science-fiction',
            'media_type': 'audiobook',
        }
        html = r'''goodreads.com/book/show/12345-the-book?from_choice=false\">The Book<\\/a><\\/h2>\\n<div>\\n by <a class=\\"authorName\\" href=\\"/author/show/1.Jane_Doe\\">Jane Doe<\\/a>\\n<\\/div>'''
        html = html.replace(chr(92) * 2, chr(92)).replace(chr(92) * 2, chr(92))
        match = re.search(_GOODREADS_BOOK_RE, html)

        result = _candidate_from_goodreads_match(match, html, source, 1)

        assert result is not None
        assert result['author'] == 'Jane Doe'
        assert result['source_uid'] == 'goodreads:12345'

    def test_goodreads_navigation_link_is_rejected_after_resolution(self):
        from books_recommender.source_discovery import _candidate_from_anchor

        source = {
            'name': 'goodreads-science-fiction',
            'url': 'https://www.goodreads.com/genres/science-fiction',
            'media_type': 'audiobook',
        }
        result = _candidate_from_anchor(
            {'href': '/book/popular_by_date/2026/8', 'text': 'New Releases'},
            '',
            '<a href="/book/popular_by_date/2026/8">New Releases</a>',
            source,
            1,
        )
        assert result is None

    def test_atom_nested_author_is_normalized(self):
        from books_recommender.feeds import normalized_feed_items

        xml = b'''<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Atom Book</title>
            <id>tag:example,2026:1</id>
            <link href="https://example.com/atom-book" />
            <author><name>Jane Doe</name></author>
            <summary>A book.</summary>
            <updated>2026-08-29T00:00:00Z</updated>
          </entry>
        </feed>'''
        with patch('books_recommender.feeds.fetch_url', return_value=(xml, 'application/atom+xml')):
            items = normalized_feed_items('https://example.com/feed.atom')

        assert items[0]['author'] == 'Jane Doe'
        assert items[0]['title'] == 'Atom Book'

    def test_openlibrary_title_lookup_requires_exact_title_and_prefers_author(self):
        from books_recommender.enrichment import _fetch_openlibrary_title

        payload = {
            'docs': [
                {'title': 'Violentia', 'author_name': ['Wrong Author'], 'language': ['eng'], 'edition_count': 99},
                {'title': 'Violentiae', 'author_name': ['Adam Freeland'], 'language': ['eng'], 'edition_count': 1},
            ]
        }
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(payload).encode()
        with patch('books_recommender.enrichment.urllib.request.urlopen', return_value=response):
            result = _fetch_openlibrary_title('Violentiae', delay=0)

        assert result is not None
        assert result['author'] == 'Adam Freeland'

    def test_openlibrary_exact_title_prefers_newer_exact_work(self):
        from books_recommender.enrichment import _fetch_openlibrary_title

        payload = {'docs': [
            {'title': 'Not Till We Are Lost', 'author_name': ['William Wenthe'], 'language': ['eng'], 'first_publish_year': 2003},
            {'title': 'Not Till We Are Lost', 'author_name': ['Dennis E. Taylor'], 'language': ['eng'], 'first_publish_year': 2024},
        ]}
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(payload).encode()
        with patch('books_recommender.enrichment.urllib.request.urlopen', return_value=response):
            result = _fetch_openlibrary_title('Not Till We Are Lost (Bobiverse, #5)', delay=0)

        assert result is not None
        assert result['author'] == 'Dennis E. Taylor'

    def test_partial_goodreads_result_walks_isbn_fallback_for_author(self):
        from books_recommender.enrichment import enrich_book_metadata

        goodreads_result = {
            'pages': 371,
            'genres': ['Fantasy'],
            'isbn13': '9780593820261',
        }
        isbn_result = {
            'pages': None,
            'genres': [],
            'author': 'Matt Dinniman',
            'isbn13': '9780593820261',
        }
        with patch('books_recommender.enrichment._fetch_goodreads', return_value=goodreads_result):
            with patch('books_recommender.enrichment._fetch_openlibrary_isbn', return_value=isbn_result) as isbn_mock:
                with patch('books_recommender.enrichment._fetch_openlibrary_title') as title_mock:
                    result = enrich_book_metadata(
                        title="Carl's Doomsday Scenario",
                        goodreads_id='212393364',
                        delay=0,
                    )

        assert result['author'] == 'Matt Dinniman'
        assert result['author_provider'] == 'openlibrary-isbn'
        assert result['provider'] == 'goodreads+openlibrary-isbn'
        assert result['fallback_used'] is True
        isbn_mock.assert_called_once_with('9780593820261', delay=0)
        title_mock.assert_not_called()

    def test_historical_repair_uses_goodreads_author_and_merges_duplicates(self):
        from books_recommender.cli import _repair_historical_goodreads_candidates
        from books_recommender.db import connect, init_db

        conn = connect(':memory:')
        init_db(conn)
        for uid, title in (
            ('anchor:1:book', 'A Good Book'),
            ('goodreads:12345:8', 'A Good Book by Jane Doe'),
        ):
            conn.execute(
                '''INSERT INTO candidates (source, source_uid, title, author, url, media_type, raw_json, status)
                   VALUES (?, ?, ?, '', ?, 'audiobook', ?, 'excluded')''',
                ('goodreads-science-fiction', uid, title,
                 'https://www.goodreads.com/book/show/12345-a-good-book', '{}'),
            )
        conn.commit()
        cfg = {
            'embeddings': {'enabled': False},
            'recommendation': {'minimum_score': 0, 'banned_format_terms': []},
            'source_weights': {},
        }
        metadata = {
            'author': 'Jane Doe',
            'author_provider': 'goodreads',
            'provider': 'goodreads',
            'providers': ['goodreads'],
        }
        with patch('books_recommender.cli.enrich_book_metadata', return_value=metadata):
            stats = _repair_historical_goodreads_candidates(conn, cfg, delay=0)

        row = conn.execute('SELECT title, author, source_uid, status FROM candidates').fetchone()
        assert stats['repaired_rows'] == 2
        assert stats['duplicate_rows_removed'] == 1
        assert stats['reopened'] == 1
        assert row['title'] == 'A Good Book'
        assert row['author'] == 'Jane Doe'
        assert row['source_uid'] == 'goodreads:12345'
        assert row['status'] == 'new'

    def test_upsert_candidate_does_not_erase_existing_author(self):
        from books_recommender.db import connect, init_db, upsert_candidate

        conn = connect(':memory:')
        init_db(conn)
        base = {
            'source': 'test', 'source_uid': 'goodreads:1', 'title': 'Book',
            'author': 'Known Author', 'url': '', 'raw': {}, 'status': 'new',
        }
        upsert_candidate(conn, base)
        upsert_candidate(conn, {**base, 'author': ''})

        row = conn.execute('SELECT author FROM candidates WHERE source_uid=?', ('goodreads:1',)).fetchone()
        assert row['author'] == 'Known Author'

    def test_field_aware_enrichment_repairs_author_with_existing_metadata(self):
        from books_recommender.cli import _enrich_candidates

        conn = TestCLIDigest()._make_db()
        conn.execute(
            '''INSERT INTO candidates (
                source, source_uid, title, author, url, cover_url, media_type,
                published_at, description, raw_json, score, score_breakdown,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
            (
                'goodreads-science-fiction', 'goodreads:123', 'Known Title', '',
                'https://www.goodreads.com/book/show/123-known-title', '', 'audiobook',
                '2026-01-01T00:00:00Z', '', json.dumps({'pages': 320, 'genres': ['Fantasy'], 'isbn13': '9780000000000'}),
                None, None, 'new',
            ),
        )
        conn.commit()
        row = conn.execute('SELECT * FROM candidates WHERE source_uid=?', ('goodreads:123',)).fetchone()
        fake_meta = {
            'provider': 'openlibrary-isbn',
            'providers': ['openlibrary-isbn'],
            'author': 'Recovered Author',
            'author_provider': 'openlibrary-isbn',
        }
        with patch('books_recommender.cli.enrich_book_metadata', return_value=fake_meta) as enrich_mock:
            stats = _enrich_candidates(conn, [row], {'enrichment': {'enabled': True, 'delay': 0, 'max_per_run': 5}})

        assert stats['enriched'] == 1
        assert stats['total_checked'] == 1
        assert conn.execute('SELECT author FROM candidates WHERE source_uid=?', ('goodreads:123',)).fetchone()['author'] == 'Recovered Author'
        enrich_mock.assert_called_once()

    def test_discord_digest_records_quality_counts(self):
        from books_recommender.discord_workflow import _build_digest_candidates

        conn = TestCLIDigest()._make_db()
        candidate = {
            'id': 7, 'title': 'Unresolved Book', 'author': '', 'source': 'test',
            'url': '', 'cover_url': '', 'media_type': 'audiobook', 'status': 'new',
        }
        scored = {'score': 80}
        with patch('books_recommender.cli._score_pending', return_value=(None, [(candidate, scored)])):
            with patch('books_recommender.cli._candidate_match_keys', return_value=set()):
                rows, alert, total = _build_digest_candidates(conn, {'recommendation': {}}, 10)

        event = conn.execute("SELECT kind, payload_json FROM events ORDER BY id DESC LIMIT 1").fetchone()
        payload = json.loads(event['payload_json'])
        assert rows == []
        assert alert is None
        assert total == 0
        assert event['kind'] == 'discord_digest_quality'
        assert payload['scored'] == 1
        assert payload['quality_filtered'] == {'missing author': 1}

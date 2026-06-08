from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import upsert_book, normalize_book_key
from .text import parse_note, slugify

EXCLUDED_FILENAMES = {'Weekly Picks.md', 'Recommendations.md'}


def iter_book_notes(vault_path: str) -> list[Path]:
    root = Path(vault_path)
    if not root.exists():
        return []
    out: list[Path] = []
    for path in root.rglob('*.md'):
        if path.name in EXCLUDED_FILENAMES:
            continue
        out.append(path)
    return sorted(out)


def import_obsidian(conn, vault_path: str) -> dict[str, int]:
    imported = 0
    skipped = 0
    existing_keys = {
        normalize_book_key(str(row['title']), str(row['author'] or ''))
        for row in conn.execute('SELECT title, author FROM books')
    }
    for path in iter_book_notes(vault_path):
        note = parse_note(path)
        meta = note['meta']
        if not note['title'] or (not note['author'] and not note['rating'] and not meta.get('goodreads_id')):
            skipped += 1
            continue
        if meta.get('type') in {'analysis', 'tracking'}:
            skipped += 1
            continue
        row = {
            'source': 'obsidian',
            'source_uid': note['uid'],
            'title': note['title'],
            'author': note.get('author', ''),
            'rating': note.get('rating'),
            'goodreads_id': note.get('goodreads_id'),
            'storygraph_id': note.get('storygraph_id'),
            'isbn': note.get('isbn'),
            'note_path': str(path),
            'tags': note.get('tags', []),
            'themes': note.get('themes', []),
            'summary': note.get('summary', ''),
            'review': note.get('review', ''),
            'body': note.get('body', ''),
            'raw': note.get('meta', {}),
        }
        key = normalize_book_key(str(note['title']), str(note.get('author', '')))
        if key in existing_keys:
            skipped += 1
            continue
        upsert_book(conn, row)
        existing_keys.add(key)
        imported += 1
    return {'imported': imported, 'skipped': skipped}


def book_text(row: dict[str, Any]) -> str:
    parts = [
        row.get('title', ''),
        row.get('author', ''),
        row.get('summary', ''),
        row.get('review', ''),
        ' '.join(row.get('tags', []) if isinstance(row.get('tags'), list) else []),
        ' '.join(row.get('themes', []) if isinstance(row.get('themes'), list) else []),
        row.get('body', ''),
    ]
    return ' '.join(p for p in parts if p)

#!/usr/bin/env python3
"""One-time bulk embedding script using remote Ollama GPU."""
import sqlite3
import json
import urllib.request
import urllib.error
import time
from pathlib import Path

# Configuration
REMOTE_OLLAMA_HOST = "http://192.168.29.159:11434"
MODEL = "qwen3-embedding:4b"
EMBEDDING_DIMS = 2560
BATCH_SIZE = 8000
REQUEST_TIMEOUT = 120
DB_PATH = "/workspace/books_recommender/data/books.db"

def get_books_to_embed(conn):
    """Get all books that need embedding."""
    embedded = set()
    for row in conn.execute('SELECT entity_id FROM embeddings WHERE entity_type = "book" AND model = ?', (MODEL,)):
        embedded.add(row[0])
    
    books = []
    for row in conn.execute('SELECT id, title, author, summary, review, tags_json, themes_json FROM books'):
        if row[0] not in embedded:
            books.append({
                'id': row[0],
                'title': row[1],
                'author': row[2],
                'summary': row[3] or '',
                'review': row[4] or '',
                'tags': json.loads(row[5]) if row[5] else [],
                'themes': json.loads(row[6]) if row[6] else [],
            })
    
    return books

def get_candidates_to_embed(conn):
    """Get all candidates that need embedding."""
    embedded = set()
    for row in conn.execute('SELECT entity_id FROM embeddings WHERE entity_type = "candidate" AND model = ?', (MODEL,)):
        embedded.add(row[0])
    
    candidates = []
    for row in conn.execute('SELECT id, title, author, description, raw_json FROM candidates'):
        if row[0] not in embedded:
            # Try to extract tags/themes from raw_json if available
            tags = []
            themes = []
            if row[4]:
                try:
                    raw = json.loads(row[4])
                    tags = raw.get('tags', [])
                    themes = raw.get('themes', [])
                except:
                    pass
            
            candidates.append({
                'id': row[0],
                'title': row[1],
                'author': row[2],
                'description': row[3] or '',
                'tags': tags,
                'themes': themes,
            })
    
    return candidates

def build_book_text(book):
    """Build text representation for embedding."""
    parts = [
        book['title'],
        book['author'],
        book['summary'],
        book['review'],
        ' '.join(book['tags']),
        ' '.join(book['themes']),
    ]
    return ' '.join(p for p in parts if p)

def build_candidate_text(candidate):
    """Build text representation for embedding."""
    parts = [
        candidate['title'],
        candidate['author'],
        candidate['description'],
        ' '.join(candidate['tags']),
        ' '.join(candidate['themes']),
    ]
    return ' '.join(p for p in parts if p)

def embed_batch(texts):
    """Send batch to remote Ollama for embedding."""
    payload = json.dumps({
        'model': MODEL,
        'input': texts,
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"{REMOTE_OLLAMA_HOST}/api/embed",
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('embeddings', [])
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return []

def store_embedding(conn, entity_type, entity_id, vector):
    """Store embedding in database."""
    import hashlib
    text_hash = hashlib.sha256(f"{MODEL}\0{entity_id}".encode()).hexdigest()
    
    import array
    vec_array = array.array('f', vector)
    vec_bytes = vec_array.tobytes()
    
    conn.execute('''
        INSERT OR REPLACE INTO embeddings 
        (entity_type, entity_id, model, text_hash, dim, vector_blob)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (entity_type, entity_id, MODEL, text_hash, EMBEDDING_DIMS, vec_bytes))

def embed_items(conn, items, entity_type, build_text_fn):
    """Embed a list of items."""
    total = len(items)
    if total == 0:
        print(f"No {entity_type}s to embed")
        return 0, 0
    
    embedded = 0
    errors = 0
    
    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        texts = [build_text_fn(b) for b in batch]
        
        print(f"Embedding {entity_type} batch {i//BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} items)...")
        
        vectors = embed_batch(texts)
        
        if len(vectors) != len(batch):
            print(f"Warning: Expected {len(batch)} vectors, got {len(vectors)}")
            errors += len(batch)
            continue
        
        for item, vector in zip(batch, vectors):
            try:
                store_embedding(conn, entity_type, item['id'], vector)
                embedded += 1
            except Exception as e:
                print(f"Error storing embedding for {entity_type} {item['id']}: {e}")
                errors += 1
        
        conn.commit()
        print(f"  ✓ Embedded {embedded}/{total} {entity_type}s")
    
    return embedded, errors

def main():
    conn = sqlite3.connect(DB_PATH)
    
    # Embed books
    print("Fetching books to embed...")
    books = get_books_to_embed(conn)
    print(f"Found {len(books)} books to embed")
    books_embedded, books_errors = embed_items(conn, books, 'book', build_book_text)
    
    # Embed candidates
    print("\nFetching candidates to embed...")
    candidates = get_candidates_to_embed(conn)
    print(f"Found {len(candidates)} candidates to embed")
    candidates_embedded, candidates_errors = embed_items(conn, candidates, 'candidate', build_candidate_text)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Books: {books_embedded} embedded, {books_errors} errors")
    print(f"  Candidates: {candidates_embedded} embedded, {candidates_errors} errors")
    print(f"  Total: {books_embedded + candidates_embedded} embedded")

if __name__ == '__main__':
    main()

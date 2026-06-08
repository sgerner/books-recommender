from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .db import get_embeddings_map
from .embeddings import blob_to_vector, candidate_text, cosine_similarity, mean_vector as dense_mean_vector, text_hash as embedding_text_hash
from .obsidian import book_text
from .text import tokenize


def _tfidf_vectors(texts: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    docs = [tokenize(t) for t in texts]
    n_docs = max(len(docs), 1)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}
    vecs: list[dict[str, float]] = []
    for doc in docs:
        counts = Counter(doc)
        total = sum(counts.values()) or 1
        vec = {term: (count / total) * idf.get(term, 1.0) for term, count in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append({k: v / norm for k, v in vec.items()})
    return vecs, idf


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _mean_vector(vectors: list[dict[str, float]], weights: list[float] | None = None) -> dict[str, float]:
    if not vectors:
        return {}
    acc: defaultdict[str, float] = defaultdict(float)
    total = 0.0
    for i, vec in enumerate(vectors):
        w = weights[i] if weights else 1.0
        total += w
        for k, v in vec.items():
            acc[k] += v * w
    if total == 0:
        return {}
    out = {k: v / total for k, v in acc.items()}
    norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
    return {k: v / norm for k, v in out.items()}


def _top_terms(rows: list[dict[str, Any]], field: str, min_rating: int | None = None, max_rating: int | None = None) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        rating = row.get('rating')
        if min_rating is not None and (rating is None or rating < min_rating):
            continue
        if max_rating is not None and (rating is None or rating > max_rating):
            continue
        for token in tokenize(' '.join(map(str, row.get(field, []))) if isinstance(row.get(field), list) else str(row.get(field, ''))):
            c[token] += 1
    return c


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _read_date(row: dict[str, Any]) -> datetime | None:
    raw = row.get('raw')
    if raw is None and row.get('raw_json'):
        try:
            raw = json.loads(row.get('raw_json') or '{}')
        except Exception:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    for key in ('date_read', 'read_at', 'user_read_at', 'date_added'):
        dt = _parse_date(raw.get(key))
        if dt:
            return dt
    for key in ('updated_at', 'created_at'):
        dt = _parse_date(row.get(key))
        if dt:
            return dt
    return None


def _time_decay_weight(row: dict[str, Any], half_life_years: float = 4.0, floor: float = 0.15) -> float:
    dt = _read_date(row)
    if not dt:
        return 1.0
    now = datetime.now(timezone.utc)
    years = max(0.0, (now - dt.astimezone(timezone.utc)).days / 365.25)
    return max(floor, math.exp(-years * math.log(2) / max(half_life_years, 0.1)))


def _rating_signal(rating: Any) -> float:
    try:
        r = int(rating)
    except Exception:
        return 0.0
    # Full 1-5 gradient centered on 3-star neutrality.
    return {1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.25}.get(r, 0.0)



_DEFAULT_BANNED_FORMAT_TERMS = (
    'graphic novel',
    'graphic-novel',
    'graphic novels',
    'graphic-novels',
    'manga',
    'manhwa',
    'manhua',
    'webcomic',
    'web comics',
    'web-comic',
)

_SERIES_HINT_RE = re.compile(
    r"\b([A-Z][\w\'’&-]*(?:\s+[A-Z][\w\'’&-]*){0,5})\s+(?:series|installment|installments|book|books|novel|cycle)\b"
)


def _candidate_search_blob(candidate: dict[str, Any]) -> str:
    raw = candidate.get('raw') or {}
    if not isinstance(raw, dict):
        raw = {}
    parts = [
        candidate.get('title', ''),
        candidate.get('author', ''),
        candidate.get('description', ''),
        candidate.get('media_type', ''),
        json.dumps(raw, ensure_ascii=False, sort_keys=True),
    ]
    return ' '.join(part for part in parts if part)


def _normalize_series_name(value: Any) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return ''
    text = re.sub(r'\s*[,;:\-–—]?\s*#\d+[a-z]?.*$', '', text)
    text = re.sub(r'\s*\(\s*#\d+[a-z]?\s*\)\s*$', '', text)
    text = re.sub(r'\s*\([^)]*\)\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ,;:-')
    if text in {'', 'a', 'an', 'the', 'none', 'standalone', 'unknown'}:
        return ''
    return text

def _series_terms_from_text(text: str) -> set[str]:
    terms: set[str] = set()
    if not text:
        return terms
    for match in _SERIES_HINT_RE.finditer(text):
        term = _normalize_series_name(match.group(1))
        if term:
            terms.add(term)
    return terms


def _series_terms_from_value(value: Any) -> set[str]:
    terms: set[str] = set()
    if value is None:
        return terms
    if isinstance(value, list):
        for item in value:
            term = _normalize_series_name(item)
            if term:
                terms.add(term)
        return terms
    term = _normalize_series_name(value)
    if term:
        terms.add(term)
    return terms


def _series_terms_from_parenthetical_title(title: str) -> set[str]:
    terms: set[str] = set()
    if not title:
        return terms
    for chunk in re.findall(r'\(([^)]{1,120})\)', title):
        lower = chunk.lower()
        if '#' not in chunk and not any(word in lower for word in ('series', 'book', 'novel', 'cycle')):
            continue
        part = re.split(r'[,;#]', chunk, maxsplit=1)[0]
        term = _normalize_series_name(part)
        if term:
            terms.add(term)
    return terms


def _candidate_series_terms(candidate: dict[str, Any]) -> set[str]:
    raw = candidate.get('raw') or {}
    if not isinstance(raw, dict):
        raw = {}
    terms: set[str] = set()
    for key in ('series', 'series_name'):
        terms.update(_series_terms_from_value(raw.get(key)))
    terms.update(_series_terms_from_parenthetical_title(candidate.get('title', '')))
    terms.update(_series_terms_from_text(candidate.get('description', '')))
    return terms

def _row_series_terms(row: dict[str, Any]) -> set[str]:
    raw = row.get('raw')
    if not isinstance(raw, dict):
        raw = {}
        raw_json = row.get('raw_json')
        if raw_json:
            try:
                raw = json.loads(raw_json)
            except Exception:
                raw = {}
    return _candidate_series_terms({
        'title': row.get('title', ''),
        'description': ' '.join(part for part in [row.get('summary', ''), row.get('review', '')] if part),
        'raw': raw,
    })


def candidate_is_banned(candidate: dict[str, Any], recommendation_cfg: dict[str, Any] | None = None) -> bool:
    rec_cfg = recommendation_cfg or {}
    terms = [str(term).strip().lower() for term in rec_cfg.get('banned_format_terms', _DEFAULT_BANNED_FORMAT_TERMS)]
    blob = _candidate_search_blob(candidate).lower()
    return any(term and term in blob for term in terms)

def build_profile(rows: list[dict[str, Any]], conn=None, embedding_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    positive = [r for r in rows if _rating_signal(r.get('rating')) > 0]
    negative = [r for r in rows if _rating_signal(r.get('rating')) < 0]
    all_texts = [book_text(r) for r in positive + negative]
    vecs, idf = _tfidf_vectors(all_texts if all_texts else [r.get('title', '') for r in rows])
    pos_vecs = vecs[: len(positive)]
    neg_vecs = vecs[len(positive):]
    pos_weights = [abs(_rating_signal(r.get('rating'))) * _time_decay_weight(r) for r in positive]
    neg_weights = [abs(_rating_signal(r.get('rating'))) * _time_decay_weight(r) for r in negative]
    profile = {
        'idf': idf,
        'positive_centroid': _mean_vector(pos_vecs, pos_weights),
        'negative_centroid': _mean_vector(neg_vecs, neg_weights),
        'positive_books': positive,
        'negative_books': negative,
        'pos_authors': Counter((r.get('author') or '').lower() for r in positive if r.get('author')),
        'neg_authors': Counter((r.get('author') or '').lower() for r in negative if r.get('author')),
        'pos_themes': _top_terms(positive, 'themes'),
        'neg_themes': _top_terms(negative, 'themes'),
        'pos_tags': _top_terms(positive, 'tags'),
        'neg_tags': _top_terms(negative, 'tags'),
        'pos_series': Counter(),
        'neg_series': Counter(),
        'embedding_model': None,
        'embedding_weight': 0.0,
        'positive_embedding_centroid': [],
        'negative_embedding_centroid': [],
        'positive_embedding_vecs': [],
        'negative_embedding_vecs': [],
    }
    profile['positive_vecs'] = pos_vecs
    profile['negative_vecs'] = neg_vecs

    for row in positive:
        weight = abs(_rating_signal(row.get('rating'))) * _time_decay_weight(row)
        for series in _row_series_terms(row):
            profile['pos_series'][series] += weight
    for row in negative:
        weight = abs(_rating_signal(row.get('rating'))) * _time_decay_weight(row)
        for series in _row_series_terms(row):
            profile['neg_series'][series] += weight

    embedding_cfg = embedding_cfg or {}
    if conn and embedding_cfg.get('enabled', True):
        model = embedding_cfg.get('model') or 'qwen3-embedding:4b'
        book_ids = [int(r['id']) for r in positive + negative if r.get('id') is not None]
        emb_map = get_embeddings_map(conn, 'book', model, book_ids)
        pos_embs = []
        neg_embs = []
        pos_emb_weights = []
        neg_emb_weights = []
        for r in positive:
            emb = emb_map.get(int(r['id'])) if r.get('id') is not None else None
            if emb:
                pos_embs.append(blob_to_vector(emb['vector_blob']))
                pos_emb_weights.append(abs(_rating_signal(r.get('rating'))) * _time_decay_weight(r))
        for r in negative:
            emb = emb_map.get(int(r['id'])) if r.get('id') is not None else None
            if emb:
                neg_embs.append(blob_to_vector(emb['vector_blob']))
                neg_emb_weights.append(abs(_rating_signal(r.get('rating'))) * _time_decay_weight(r))
        profile['embedding_model'] = model
        profile['embedding_weight'] = float(embedding_cfg.get('book_weight', 0.45))
        profile['positive_embedding_centroid'] = dense_mean_vector(pos_embs, pos_emb_weights)
        profile['negative_embedding_centroid'] = dense_mean_vector(neg_embs, neg_emb_weights)
        profile['positive_embedding_vecs'] = pos_embs
        profile['negative_embedding_vecs'] = neg_embs
    return profile

def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    vec = {term: (count / total) * idf.get(term, 1.0) for term, count in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def score_candidate(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    source_weight: float = 0.0,
    explain: bool = True,
    recommendation_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec_cfg = recommendation_cfg or {}
    if candidate_is_banned(candidate, rec_cfg):
        return {
            'score': 0,
            'raw': -1.0,
            'pos_sim': 0.0,
            'neg_sim': 0.0,
            'emb_pos_sim': 0.0,
            'emb_neg_sim': 0.0,
            'reasons': ['blocked format: graphic novel'],
            'similar_books': [],
            'semantic_books': [],
        }

    text_parts = [
        candidate.get('title', ''),
        candidate.get('author', ''),
        candidate.get('description', ''),
        ' '.join(candidate.get('tags', []) if isinstance(candidate.get('tags'), list) else []),
        ' '.join(candidate.get('themes', []) if isinstance(candidate.get('themes'), list) else []),
    ]
    text = ' '.join(p for p in text_parts if p)
    v = vectorize(text, profile['idf'])
    pos_sim = _dot(v, profile['positive_centroid'])
    neg_sim = _dot(v, profile['negative_centroid'])
    emb_vec = candidate.get('embedding') or []
    emb_pos_sim = 0.0
    emb_neg_sim = 0.0
    semantic_raw = 0.0
    semantic = []
    if emb_vec and profile.get('positive_embedding_centroid'):
        emb_pos_sim = cosine_similarity(emb_vec, profile.get('positive_embedding_centroid') or [])
        emb_neg_sim = cosine_similarity(emb_vec, profile.get('negative_embedding_centroid') or [])
        semantic_raw = (0.85 * emb_pos_sim) - (0.60 * emb_neg_sim)
        if explain:
            for book, vec in zip(profile['positive_books'], profile.get('positive_embedding_vecs', [])):
                sim = cosine_similarity(emb_vec, vec)
                if sim > 0:
                    semantic.append((sim, book))
            semantic.sort(reverse=True, key=lambda x: x[0])
    author = (candidate.get('author') or '').lower().strip()
    author_bonus = 0.15 if author and profile['pos_authors'].get(author) else 0.0
    neg_author_penalty = 0.08 if author and profile['neg_authors'].get(author) else 0.0
    pos_theme_overlap = len(set(map(str.lower, candidate.get('themes', []))) & set(profile['pos_themes']))
    neg_theme_overlap = len(set(map(str.lower, candidate.get('themes', []))) & set(profile['neg_themes']))
    theme_bonus = 0.03 * pos_theme_overlap - 0.05 * neg_theme_overlap
    candidate_series = _candidate_series_terms(candidate)
    neg_series_overlap = sorted(candidate_series & set(profile.get('neg_series', {}).keys()))
    pos_series_overlap = sorted(candidate_series & set(profile.get('pos_series', {}).keys()))
    series_negative_penalty = float(rec_cfg.get('series_negative_penalty', 0.22))
    series_positive_bonus = float(rec_cfg.get('series_positive_bonus', 0.05))
    series_bonus = 0.0
    if neg_series_overlap:
        neg_series_weight = sum(float(profile['neg_series'][series]) for series in neg_series_overlap)
        series_bonus -= series_negative_penalty * min(neg_series_weight, 2.0)
    if pos_series_overlap:
        pos_series_weight = sum(float(profile['pos_series'][series]) for series in pos_series_overlap)
        series_bonus += series_positive_bonus * min(pos_series_weight, 2.0)
    embed_weight = profile.get('embedding_weight', 0.0) if emb_vec and profile.get('positive_embedding_centroid') else 0.0
    tfidf_raw = (0.75 * pos_sim) - (0.55 * neg_sim)
    raw = ((1 - embed_weight) * tfidf_raw) + (embed_weight * semantic_raw) + author_bonus - neg_author_penalty + theme_bonus + series_bonus + source_weight
    score = max(0, min(100, round(50 + raw * 50)))
    similar = []
    if explain:
        for book, vec in zip(profile['positive_books'], profile.get('positive_vecs', [])):
            sim = _dot(v, vec)
            if sim > 0:
                similar.append((sim, book))
        similar.sort(reverse=True, key=lambda x: x[0])
    reasons = []
    if similar:
        reasons.append('similar to ' + ', '.join(f"{b.get('title')}" for _, b in similar[:3]))
    if semantic:
        reasons.append('semantic match to ' + ', '.join(f"{b.get('title')}" for _, b in semantic[:3]))
    if pos_theme_overlap:
        reasons.append(f"theme overlap: {', '.join(sorted(set(map(str.lower, candidate.get('themes', []))) & set(profile['pos_themes']))[:4])}")
    if author_bonus:
        reasons.append('same author cluster as 5-star books')
    if neg_series_overlap:
        reasons.append('series match to low-rated books: ' + ', '.join(neg_series_overlap[:3]))
    elif pos_series_overlap:
        reasons.append('series match to liked books: ' + ', '.join(pos_series_overlap[:3]))
    if source_weight:
        reasons.append(f'source weight +{source_weight:.2f}')
    return {
        'score': score,
        'raw': raw,
        'pos_sim': pos_sim,
        'neg_sim': neg_sim,
        'emb_pos_sim': emb_pos_sim,
        'emb_neg_sim': emb_neg_sim,
        'reasons': reasons,
        'similar_books': [{'title': b.get('title'), 'author': b.get('author'), 'rating': b.get('rating'), 'similarity': round(sim, 3)} for sim, b in similar[:5]],
        'semantic_books': [{'title': b.get('title'), 'author': b.get('author'), 'rating': b.get('rating'), 'similarity': round(sim, 3)} for sim, b in semantic[:5]],
    }


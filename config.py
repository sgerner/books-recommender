from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    'db_path': '/workspace/books_recommender/data/books.db',
    'vault_path': '/workspace/obsidian/Media/Books',
    'goodreads_rss_url': 'https://www.goodreads.com/review/list_rss/5506302?shelf=read',
    'sources': [],
    'sources_inbox_path': 'Media/Book Sources Inbox.md',
    'recommendation': {
        'top_n': 10,
        'strong_threshold': 72,
        'medium_threshold': 58,
        'auto_add_threshold': 80,
    },
    'librarr': {
        'url': 'http://librarr:5050',
        'api_key': None,
        'api_key_env': 'LIBRARR_API_KEY',
        'wishlist_media_type': 'audiobook',
    },
    'nyt': {
        'api_key': None,
        'api_key_env': 'NYT_API_KEY',
    },
    'embeddings': {
        'enabled': True,
        'model': 'qwen3-embedding:4b',
        'base_url': 'http://ollama:11434',
        'batch_size': 16,
        'candidate_weight': 0.4,
        'book_weight': 0.45,
    },
    'discord': {
        'enabled': False,
        'token_env': 'DISCORD_BOT_TOKEN',
        'channel_id': None,
        'post_hour_utc': 15,
        'post_weekday': 1,
    },
    'source_weights': {
        'goodreads': 0.35,
        'nyt': 0.55,
        'blog': 0.45,
        'rss': 0.4,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def load_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    config_path = Path(path or os.environ.get('BOOKS_RECOMMENDER_CONFIG', '/workspace/books_recommender/config.json'))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        cfg = deep_merge(cfg, load_json(config_path))
    secrets_path = config_path.with_name('secrets.json')
    if secrets_path.exists():
        cfg = deep_merge(cfg, load_json(secrets_path))

    env = os.environ
    for env_name, target in [
        ('BOOKS_RECOMMENDER_DB_PATH', ('db_path',)),
        ('BOOKS_RECOMMENDER_VAULT_PATH', ('vault_path',)),
        ('GOODREADS_RSS_URL', ('goodreads_rss_url',)),
        ('LIBRARR_URL', ('librarr', 'url')),
        ('LIBRARR_API_KEY', ('librarr', 'api_key')),
        ('NYT_API_KEY', ('nyt', 'api_key')),
        ('NYTIMES_API_KEY', ('nyt', 'api_key')),
        ('DISCORD_CHANNEL_ID', ('discord', 'channel_id')),
        ('DISCORD_BOT_TOKEN', ('discord', 'token')),
    ]:
        value = env.get(env_name)
        if value:
            cur = cfg
            for key in target[:-1]:
                cur = cur.setdefault(key, {})
            cur[target[-1]] = value

    return cfg


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def resolve_librarr_api_key(cfg: Dict[str, Any]) -> str | None:
    if cfg.get('librarr', {}).get('api_key'):
        return cfg['librarr']['api_key']
    env_name = cfg.get('librarr', {}).get('api_key_env', 'LIBRARR_API_KEY')
    return os.environ.get(env_name)


def resolve_nyt_api_key(cfg: Dict[str, Any]) -> str | None:
    key = cfg.get('nyt', {}).get('api_key')
    if key:
        return key
    env_name = cfg.get('nyt', {}).get('api_key_env', 'NYT_API_KEY')
    return os.environ.get(env_name) or os.environ.get('NYTIMES_API_KEY')


def resolve_discord_token(cfg: Dict[str, Any]) -> str | None:
    token = cfg.get('discord', {}).get('token')
    if token:
        return token
    env_name = cfg.get('discord', {}).get('token_env', 'DISCORD_BOT_TOKEN')
    return os.environ.get(env_name)

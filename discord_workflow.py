from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .config import load_config, resolve_discord_token, resolve_librarr_api_key
from .db import (
    connect,
    get_pending_discord_messages,
    init_db,
    record_feedback,
    set_candidate_status,
    update_discord_message_status,
    upsert_discord_message,
)
from .librarr import LibrarrClient

DISCORD_API_BASE = "https://discord.com/api/v10"
THUMBS_UP = "👍"
THUMBS_DOWN = "👎"


class DiscordApiError(RuntimeError):
    pass


def _discord_request(token: str, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{DISCORD_API_BASE}{path}"
    headers = {
        "Authorization": f"Bot {token}",
        "Accept": "application/json",
        "User-Agent": "Hermes-BooksRecommender/1.0",
    }
    data = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if not body.strip():
                return {"status": resp.status}
            if resp.headers.get_content_type() == "application/json" or body.lstrip().startswith(("{", "[")):
                return json.loads(body)
            return {"status": resp.status, "body": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        if exc.code == 429:
            retry_after = payload.get("retry_after")
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                time.sleep(float(retry_after) + 0.25)
                return _discord_request(token, method, path, json_body=json_body)
        raise DiscordApiError(f"Discord {method} {path} failed ({exc.code}): {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DiscordApiError(f"Discord {method} {path} failed: {exc}") from exc


def discord_post_message(token: str, channel_id: str, content: str) -> dict[str, Any]:
    return _discord_request(token, "POST", f"/channels/{channel_id}/messages", json_body={"content": content})


def discord_add_reaction(token: str, channel_id: str, message_id: str, emoji: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(emoji, safe="")
    return _discord_request(token, "PUT", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me")


def discord_fetch_message(token: str, channel_id: str, message_id: str) -> dict[str, Any]:
    return _discord_request(token, "GET", f"/channels/{channel_id}/messages/{message_id}")


def discord_edit_message(token: str, channel_id: str, message_id: str, content: str) -> dict[str, Any]:
    return _discord_request(token, "PATCH", f"/channels/{channel_id}/messages/{message_id}", json_body={"content": content})


def _resolve_discord_channel_id(token: str, channel_ref: str) -> str:
    ref = str(channel_ref or "").strip()
    if not ref:
        raise DiscordApiError("Discord channel reference is empty")
    if ref.isdigit():
        return ref

    normalized = ref.lstrip("#")
    guilds = _discord_request(token, "GET", "/users/@me/guilds")
    if not isinstance(guilds, list):
        raise DiscordApiError("Discord did not return a guild list")
    for guild in guilds:
        guild_id = guild.get("id") if isinstance(guild, dict) else None
        if not guild_id:
            continue
        channels = _discord_request(token, "GET", f"/guilds/{guild_id}/channels")
        if not isinstance(channels, list):
            continue
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            name = str(channel.get("name") or "")
            if name == normalized or name == ref:
                return str(channel.get("id"))
    raise DiscordApiError(f"Could not resolve Discord channel reference {channel_ref!r}")


def _reaction_count(message: dict[str, Any], emoji: str) -> int:
    for reaction in message.get("reactions") or []:
        if isinstance(reaction, dict) and reaction.get("emoji", {}).get("name") == emoji:
            try:
                return int(reaction.get("count") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _reaction_decision(message: dict[str, Any]) -> tuple[str | None, dict[str, int]]:
    counts = {
        THUMBS_UP: _reaction_count(message, THUMBS_UP),
        THUMBS_DOWN: _reaction_count(message, THUMBS_DOWN),
    }
    up = counts[THUMBS_UP]
    down = counts[THUMBS_DOWN]
    # The bot adds one of each reaction immediately after posting, so human
    # feedback is visible as count > 1.
    if up > down and up > 1:
        return "approve", counts
    if down > up and down > 1:
        return "reject", counts
    return None, counts


def _format_recommendation(candidate: dict[str, Any], scored: dict[str, Any]) -> str:
    title = candidate.get("title") or "Untitled"
    author = candidate.get("author") or "Unknown"
    score = scored.get("score", 0)
    lines = [
        f"**{title}** — {author}",
        f"Score: {score}/100",
    ]
    candidate_id = candidate.get("id")
    if candidate_id is not None:
        lines.append(f"Candidate ID: {candidate_id}")
    if candidate.get("url"):
        lines.append(f"Source: {candidate['url']}")
    if candidate.get("cover_url"):
        lines.append(f"Cover: {candidate['cover_url']}")

    reasons = scored.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Why this made the cut:")
        for reason in reasons[:3]:
            lines.append(f"- {reason}")

    similar = scored.get("similar_books") or []
    if similar:
        lines.append("")
        lines.append("Closest matches from your library:")
        for item in similar[:2]:
            lines.append(f"- {item.get('title')} — {item.get('author') or 'Unknown'} ({item.get('rating')} stars)")

    lines.append("")
    lines.append(f"React {THUMBS_UP} to approve or {THUMBS_DOWN} to reject.")
    return "\n".join(lines)


def _build_digest_candidates(conn, cfg: dict[str, Any], limit: int) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str | None, int]:
    # Weekly digest mode: enrich missing metadata, score pending items, and
    # filter out anything that already matches an existing book.
    from .cli import _candidate_match_keys, _score_pending, normalize_candidate_match_key, candidate_match_keys

    _, changed = _score_pending(conn, cfg, sync_books=False, explain=True, enrich=True)
    book_keys = _candidate_match_keys(conn)
    filtered = [
        (candidate, scored)
        for candidate, scored in changed
        if not (candidate_match_keys(candidate.get("title", ""), candidate.get("author", ""), candidate.get("source", "")) & book_keys)
    ]
    filtered.sort(key=lambda x: x[1].get("score", 0), reverse=True)
    rows = filtered[:limit]
    min_candidates = int(cfg.get("recommendation", {}).get("min_candidates", 3))
    alert = None
    if len(filtered) < min_candidates:
        alert = f"WARNING: Only {len(filtered)} candidates scored (threshold: {min_candidates}). Pipeline may need attention."
    return rows, alert, len(filtered)


def post_recommendations(
    conn,
    cfg: dict[str, Any],
    *,
    limit: int = 10,
    channel_id: str | None = None,
) -> dict[str, Any]:
    discord_cfg = cfg.get("discord", {})
    if not discord_cfg.get("enabled", False):
        raise DiscordApiError("Discord delivery is disabled in config")

    token = resolve_discord_token(cfg)
    if not token:
        raise DiscordApiError("Discord bot token is not configured")

    target_channel_ref = str(channel_id or discord_cfg.get("channel_id") or "").strip()
    if not target_channel_ref:
        raise DiscordApiError("Discord channel_id is not configured")
    target_channel = _resolve_discord_channel_id(token, target_channel_ref)

    rows, alert, total_scored = _build_digest_candidates(conn, cfg, limit)
    posted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if alert:
        print(alert)

    for candidate, scored in rows:
        if candidate.get("status") != "new":
            skipped.append({"id": candidate.get("id"), "reason": f"status={candidate.get('status')}"})
            continue

        content = _format_recommendation(candidate, scored)
        response = discord_post_message(token, target_channel, content)
        message_id = str(response.get("id") or "")
        if not message_id:
            raise DiscordApiError(f"Discord did not return a message id for candidate {candidate.get('id')}")

        discord_add_reaction(token, target_channel, message_id, THUMBS_UP)
        discord_add_reaction(token, target_channel, message_id, THUMBS_DOWN)

        upsert_discord_message(
            conn,
            candidate_id=int(candidate["id"]),
            channel_id=target_channel,
            message_id=message_id,
            content=content,
            status="posted",
        )
        set_candidate_status(conn, int(candidate["id"]), "discord_pending", note="posted to Discord for approval")
        record_feedback(conn, int(candidate["id"]), "discord_post", "", f"discord:{target_channel}")
        posted.append({
            "candidate_id": candidate.get("id"),
            "message_id": message_id,
            "title": candidate.get("title"),
            "author": candidate.get("author"),
            "score": scored.get("score"),
        })

    return {
        "channel_id": target_channel,
        "limit": limit,
        "total_scored": total_scored,
        "posted": posted,
        "skipped": skipped,
        "alert": alert,
    }


def _mark_approved(conn, cfg: dict[str, Any], *, candidate: dict[str, Any], message_row, reaction_counts: dict[str, int]) -> dict[str, Any]:
    client = LibrarrClient(cfg["librarr"]["url"], resolve_librarr_api_key(cfg))
    result = client.add_to_wishlist(candidate.get("title", ""), candidate.get("author") or "", cfg["librarr"].get("wishlist_media_type", "audiobook"))
    librarr_id = None
    if isinstance(result, dict):
        if result.get("id") is not None:
            librarr_id = str(result.get("id"))
        elif result.get("wishlist_id") is not None:
            librarr_id = str(result.get("wishlist_id"))
    if result and isinstance(result, dict) and result.get("error"):
        raise DiscordApiError(f"Librarr rejected wishlist item: {result.get('body') or result.get('error')}")

    set_candidate_status(conn, int(candidate["id"]), "approved", librarr_id=librarr_id, note="approved via Discord reaction")
    update_discord_message_status(
        conn,
        candidate_id=int(candidate["id"]),
        status="approved",
        reaction=THUMBS_UP,
        reaction_counts_json=json.dumps(reaction_counts, ensure_ascii=False),
        librarr_id=librarr_id,
    )
    record_feedback(conn, int(candidate["id"]), "approve", "discord reaction 👍", f"discord:{message_row['channel_id']}")
    return {"candidate_id": candidate.get("id"), "status": "approved", "librarr": result, "librarr_id": librarr_id}


def _mark_rejected(conn, *, candidate: dict[str, Any], message_row, reaction_counts: dict[str, int]) -> dict[str, Any]:
    set_candidate_status(conn, int(candidate["id"]), "rejected", note="rejected via Discord reaction")
    update_discord_message_status(
        conn,
        candidate_id=int(candidate["id"]),
        status="rejected",
        reaction=THUMBS_DOWN,
        reaction_counts_json=json.dumps(reaction_counts, ensure_ascii=False),
    )
    record_feedback(conn, int(candidate["id"]), "reject", "discord reaction 👎", f"discord:{message_row['channel_id']}")
    return {"candidate_id": candidate.get("id"), "status": "rejected"}


def sync_reactions(conn, cfg: dict[str, Any], *, channel_id: str | None = None) -> dict[str, Any]:
    discord_cfg = cfg.get("discord", {})
    if not discord_cfg.get("enabled", False):
        raise DiscordApiError("Discord delivery is disabled in config")

    token = resolve_discord_token(cfg)
    if not token:
        raise DiscordApiError("Discord bot token is not configured")

    pending = get_pending_discord_messages(conn)
    resolved_channel_id = None
    if channel_id:
        resolved_channel_id = _resolve_discord_channel_id(token, channel_id)
        pending = [row for row in pending if str(row["channel_id"]) == str(resolved_channel_id)]

    processed = []
    skipped = []
    errors = []

    for row in pending:
        candidate = {
            "id": row["candidate_id"],
            "title": row["title"],
            "author": row["author"],
            "url": row["url"],
            "cover_url": row["cover_url"],
            "media_type": row["media_type"],
            "published_at": row["published_at"],
            "description": row["description"],
            "raw": json.loads(row["raw_json"] or "{}"),
            "score": row["score"],
            "score_breakdown": json.loads(row["score_breakdown"] or "{}"),
            "status": row["candidate_status"],
        }
        try:
            message = discord_fetch_message(token, str(row["channel_id"]), str(row["message_id"]))
            decision, reaction_counts = _reaction_decision(message)
            if not decision:
                skipped.append({"candidate_id": candidate["id"], "reason": "no decision yet"})
                continue

            if decision == "approve":
                result = _mark_approved(conn, cfg, candidate=candidate, message_row=row, reaction_counts=reaction_counts)
                processed.append(result)
                final_text = f"{row['content']}\n\n✅ Approved and added to Librarr"
                if result.get("librarr_id"):
                    final_text += f" (ID {result['librarr_id']})"
                discord_edit_message(token, str(row["channel_id"]), str(row["message_id"]), final_text)
                continue

            result = _mark_rejected(conn, candidate=candidate, message_row=row, reaction_counts=reaction_counts)
            processed.append(result)
            final_text = f"{row['content']}\n\n❌ Rejected"
            discord_edit_message(token, str(row["channel_id"]), str(row["message_id"]), final_text)
        except Exception as exc:
            errors.append({"candidate_id": candidate["id"], "error": str(exc)})

    return {"processed": processed, "skipped": skipped, "errors": errors, "count": len(processed)}


def cmd_discord_post(args) -> None:
    cfg = load_config(args.config)
    conn = connect(cfg["db_path"])
    init_db(conn)
    result = post_recommendations(conn, cfg, limit=args.limit, channel_id=args.channel_id)
    if result.get("alert"):
        print(result["alert"])


def cmd_discord_sync_reactions(args) -> None:
    cfg = load_config(args.config)
    conn = connect(cfg["db_path"])
    init_db(conn)
    result = sync_reactions(conn, cfg, channel_id=args.channel_id)
    if result.get("count") or result.get("errors"):
        print(json.dumps(result, indent=2, ensure_ascii=False))

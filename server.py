#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from difflib import SequenceMatcher
import json
import mimetypes
import os
import re
import select
import shutil
import subprocess
import threading
import time
import uuid
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

HOST = "0.0.0.0"
PORT = 8000
INDEX_PATH = Path(__file__).with_name("index.html")
BASE_DIR = Path(__file__).parent.resolve()
SESSION_TTL_SECONDS = 60 * 60
VAAPI_DEVICE = "/dev/dri/renderD128"
VIDEO_WIDTH = 854
VIDEO_Q_CPU = "14"
LIVE_PROBESIZE = "5M"
LIVE_ANALYZEDURATION = "5M"
LIVE_AUDIO_PROBESIZE = "1M"
LIVE_AUDIO_ANALYZEDURATION = "1M"
LIVE_VIDEO_FPS = 15
H264_VIDEO_CRF = "25"
H264_VIDEO_PRESET = "veryfast"
H264_GOP = "30"
H264_TARGET_FPS = 30
LIVE_FIRST_CHUNK_TIMEOUT_SECONDS = 25.0
DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 4.0
LIVE_BUFFER_ROOT = Path("/tmp/splay_live_buffer")
LIVE_BUFFER_READY_TIMEOUT_SECONDS = 20.0
LIVE_BUFFER_MIN_SEGMENTS = 3
LIVE_BUFFER_HLS_TIME_SECONDS = 2
CHANNELS_M3U_URL = "http://192.168.0.182:9191/output/m3u"
PLEX_CONTROL_DIR = Path("/mnt/synology/misc/dev/plex_control")
PLEX_CTL_PATH = PLEX_CONTROL_DIR / "plexctl.py"
PLEX_MAPPING_PATH = PLEX_CONTROL_DIR / "plex_content_mapping.json"
PLEX_HISTORY_PATH = BASE_DIR / "plex_history.json"
PLEX_HISTORY_MAX_ITEMS = 200
PLEX_PULL_MAX_ITEMS = 80
PLEX_SYNC_STATUS = {
    "last_pull_at": None,
    "last_push_at": None,
    "last_pull_error": "",
    "last_push_error": "",
}

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
PLEX_CFG_LOCK = threading.Lock()
PLEX_HISTORY_LOCK = threading.Lock()
PLEX_SYNC_LOCK = threading.Lock()
PLEX_CFG = None
PLEX_MAPPING_CACHE = {"mtime": None, "items": []}


def load_plex_config():
    global PLEX_CFG
    with PLEX_CFG_LOCK:
        if PLEX_CFG is not None:
            return PLEX_CFG

        if not PLEX_CTL_PATH.exists():
            raise RuntimeError(f"Missing Plex config file: {PLEX_CTL_PATH}")

        text = PLEX_CTL_PATH.read_text(encoding="utf-8", errors="ignore")
        base_match = re.search(r"PLEX_BASE_URL\s*=\s*['\"]([^'\"]+)['\"]", text)
        token_match = re.search(r"PLEX_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", text)
        if not base_match or not token_match:
            raise RuntimeError("Could not parse Plex base URL/token from plexctl.py")

        PLEX_CFG = {
            "base_url": base_match.group(1).rstrip("/"),
            "token": token_match.group(1).strip(),
        }
        return PLEX_CFG


def normalized_search_text(value):
    text = re.sub(r"\(\d{4}\)", "", value.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_plex_mapping():
    mtime = PLEX_MAPPING_PATH.stat().st_mtime if PLEX_MAPPING_PATH.exists() else None
    if PLEX_MAPPING_CACHE["mtime"] == mtime:
        return PLEX_MAPPING_CACHE["items"]

    items = []
    if PLEX_MAPPING_PATH.exists():
        raw = json.loads(PLEX_MAPPING_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for title, info in raw.items():
                if not isinstance(info, dict):
                    continue
                content_id = str(info.get("content_id", "")).strip()
                if not content_id:
                    continue
                items.append({
                    "title": title,
                    "type": str(info.get("type", "movie")),
                    "content_id": content_id,
                    "year": info.get("year"),
                    "search_title": normalized_search_text(title),
                })

    PLEX_MAPPING_CACHE["mtime"] = mtime
    PLEX_MAPPING_CACHE["items"] = items
    return items


def score_match(query_norm, title_norm):
    if not query_norm or not title_norm:
        return 0
    if query_norm == title_norm:
        return 1000
    score = int(SequenceMatcher(None, query_norm, title_norm).ratio() * 100)
    if query_norm in title_norm:
        score += 120
    q_tokens = query_norm.split()
    if q_tokens and all(tok in title_norm for tok in q_tokens):
        score += 80
    return score


def _search_local_mapping(query, limit=12):
    q = normalized_search_text(query)
    if not q:
        return []
    items = load_plex_mapping()

    exact = [item for item in items if item["search_title"] == q]
    if exact:
        return [{
            "title": item["title"],
            "type": item["type"],
            "content_id": item["content_id"],
            "year": item.get("year"),
            "score": 2000,
        } for item in exact[:limit]]

    scored = []
    for item in items:
        score = score_match(q, item["search_title"])
        if score > 30:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, item in scored[:limit]:
        out.append({
            "title": item["title"],
            "type": item["type"],
            "content_id": item["content_id"],
            "year": item.get("year"),
            "score": score,
        })
    return out


def _type_from_node(node):
    node_type = (node.get("type") or "").lower()
    if node_type == "show":
        return "series"
    if node_type in ("movie", "episode", "season"):
        return node_type
    return None


def plex_search_live(query, limit=12):
    q = normalized_search_text(query)
    if not q:
        return []

    nodes = []

    # 1) Global library search
    try:
        root = plex_get_xml("/library/search?query=" + urllib.parse.quote(query))
        nodes.extend(root.findall("./Video"))
        nodes.extend(root.findall("./Directory"))
    except Exception:
        pass

    # 2) Hubs search can return items that library/search misses.
    try:
        root = plex_get_xml("/hubs/search?query=" + urllib.parse.quote(query))
        for hub in root.findall("./Hub"):
            nodes.extend(hub.findall("./Video"))
            nodes.extend(hub.findall("./Directory"))
    except Exception:
        pass

    # 3) Section-level search across movie/show libraries.
    try:
        sections_root = plex_get_xml("/library/sections")
        for section in sections_root.findall("./Directory"):
            sec_id = (section.get("key") or "").strip()
            if not sec_id:
                continue
            sec_type = (section.get("type") or "").lower()
            if sec_type not in ("movie", "show"):
                continue
            type_code = "1" if sec_type == "movie" else "2"
            path = f"/library/sections/{sec_id}/search?query={urllib.parse.quote(query)}&type={type_code}"
            try:
                sec_root = plex_get_xml(path)
                nodes.extend(sec_root.findall("./Video"))
                nodes.extend(sec_root.findall("./Directory"))
            except Exception:
                continue
    except Exception:
        pass
    results = []
    seen = set()

    for node in nodes:
        item_type = _type_from_node(node)
        if item_type is None:
            continue
        content_id = (node.get("ratingKey") or "").strip()
        if not content_id:
            continue

        title = (node.get("title") or "").strip()
        if item_type == "episode":
            gp = (node.get("grandparentTitle") or "").strip()
            ep = (node.get("title") or "").strip()
            if gp and ep:
                title = f"{gp} - {ep}"

        key = (item_type, content_id)
        if key in seen:
            continue
        seen.add(key)

        score = score_match(q, normalized_search_text(title))
        year_value = None
        raw_year = (node.get("year") or "").strip()
        if raw_year.isdigit():
            year_value = int(raw_year)
        results.append({
            "title": title or "Untitled",
            "type": item_type,
            "content_id": content_id,
            "year": year_value,
            "score": score,
        })

    if not results:
        return []

    exact = [r for r in results if normalized_search_text(r["title"]) == q]
    if exact:
        exact.sort(key=lambda r: r["title"])
        return exact[:limit]

    results.sort(key=lambda r: r["score"], reverse=True)
    return [r for r in results if r["score"] > 20][:limit]


def plex_search(query, limit=12):
    # Prefer live Plex query for completeness; fallback to local cache if unavailable.
    try:
        live_results = plex_search_live(query, limit=limit)
        if live_results:
            return live_results
    except Exception:
        pass
    return _search_local_mapping(query, limit=limit)


def plex_get_bytes(path, timeout=20):
    cfg = load_plex_config()
    sep = "&" if "?" in path else "?"
    url = f"{cfg['base_url']}{path}{sep}X-Plex-Token={urllib.parse.quote(cfg['token'])}"
    req = urllib.request.Request(url, headers={"User-Agent": "splay/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read()


def plex_get_xml(path, timeout=20):
    return ET.fromstring(plex_get_bytes(path, timeout=timeout))


def plex_get_metadata_video(content_id):
    root = plex_get_xml(f"/library/metadata/{content_id}")
    video = root.find("./Video")
    if video is None:
        return None, root
    return video, root


def parse_part_id_from_url(video_url):
    value = str(video_url or "").strip()
    match = re.search(r"/library/parts/(\d+)", value)
    if not match:
        return ""
    return match.group(1)


def extract_history_title(video):
    item_type = (video.get("type") or "").lower()
    if item_type == "episode":
        gp = (video.get("grandparentTitle") or "").strip()
        ep = (video.get("title") or "").strip()
        if gp and ep:
            return f"{gp} - {ep}"
    return (video.get("title") or "").strip() or "Untitled"


def resolve_content_id_from_part_id(part_id):
    value = str(part_id or "").strip()
    if not value:
        return ""
    try:
        root = plex_get_xml(f"/library/parts/{value}")
        part = root.find(".//Part")
        if part is None:
            return ""
        key = (part.get("key") or "").strip()
        match = re.search(r"/library/metadata/(\d+)", key)
        if match:
            return match.group(1)
    except Exception:
        return ""
    return ""


def resolve_content_id_from_title(title, item_type):
    query = str(title or "").strip()
    wanted = str(item_type or "").strip().lower()
    if not query:
        return ""
    try:
        results = plex_search(query, limit=20)
    except Exception:
        return ""
    if not results:
        return ""
    norm_q = normalized_search_text(query)
    candidates = []
    for item in results:
        cid = str(item.get("content_id") or "").strip()
        if not cid:
            continue
        typ = str(item.get("type") or "").strip().lower()
        title_norm = normalized_search_text(str(item.get("title") or ""))
        score = int(item.get("score") or 0)
        if wanted and typ == wanted:
            score += 120
        if norm_q and title_norm == norm_q:
            score += 500
        elif norm_q and norm_q in title_norm:
            score += 120
        candidates.append((score, cid))
    if not candidates:
        return ""
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def plex_sync_write_timeline(content_id, position_seconds, duration_seconds, state):
    rating_key = str(content_id or "").strip()
    if not rating_key:
        raise RuntimeError("Missing content_id")
    pos_sec = max(0.0, float(position_seconds or 0.0))
    duration_sec = float(duration_seconds or 0.0)
    if duration_sec <= 0:
        video, _ = plex_get_metadata_video(rating_key)
        if video is not None:
            try:
                duration_sec = float(video.get("duration", "0") or "0") / 1000.0
            except (TypeError, ValueError):
                duration_sec = 0.0
    duration_ms = int(max(pos_sec, duration_sec, 1.0) * 1000.0)
    position_ms = int(pos_sec * 1000.0)
    state_value = str(state or "playing").strip().lower()
    if state_value not in ("playing", "paused", "stopped", "buffering"):
        state_value = "playing"

    path = (
        "/:/timeline"
        f"?ratingKey={urllib.parse.quote(rating_key)}"
        f"&key={urllib.parse.quote('/library/metadata/' + rating_key)}"
        "&identifier=com.plexapp.plugins.library"
        f"&state={urllib.parse.quote(state_value)}"
        f"&time={position_ms}"
        f"&duration={duration_ms}"
        "&X-Plex-Client-Identifier=splay-sync"
        "&X-Plex-Product=Splay"
        "&X-Plex-Version=1.0"
        "&X-Plex-Platform=Web"
    )
    plex_get_bytes(path, timeout=12)


def plex_sync_mark_watched(content_id):
    rating_key = str(content_id or "").strip()
    if not rating_key:
        return
    path = (
        "/:/scrobble"
        f"?key={urllib.parse.quote(rating_key)}"
        "&identifier=com.plexapp.plugins.library"
    )
    plex_get_bytes(path, timeout=12)


def plex_fetch_recent_history(limit=PLEX_PULL_MAX_ITEMS):
    root = plex_get_xml("/status/sessions/history/all")
    out = []
    seen = set()
    now_ms = int(time.time() * 1000)
    for node in root.findall("./Video"):
        content_id = (node.get("ratingKey") or "").strip()
        item_type = (node.get("type") or "").strip().lower()
        if not content_id or item_type not in ("movie", "episode"):
            continue
        if content_id in seen:
            continue
        seen.add(content_id)
        try:
            video, _ = plex_get_metadata_video(content_id)
        except Exception:
            continue
        if video is None:
            continue
        part = video.find(".//Part")
        if part is None or not part.get("key"):
            continue

        duration_sec = 0.0
        offset_sec = 0.0
        try:
            duration_sec = float(video.get("duration", "0") or "0") / 1000.0
        except (TypeError, ValueError):
            duration_sec = 0.0
        try:
            offset_sec = float(video.get("viewOffset", "0") or "0") / 1000.0
        except (TypeError, ValueError):
            offset_sec = 0.0
        try:
            view_count = int(video.get("viewCount", "0") or "0")
        except (TypeError, ValueError):
            view_count = 0

        if offset_sec <= 0 and view_count <= 0:
            continue
        if offset_sec <= 0 and view_count > 0 and duration_sec > 0:
            offset_sec = duration_sec

        viewed_ts = 0
        for key in ("lastViewedAt", "viewedAt", "updatedAt"):
            raw = (video.get(key) or "").strip()
            if raw.isdigit():
                viewed_ts = int(raw)
                break
        updated_at_ms = viewed_ts * 1000 if viewed_ts > 0 else now_ms
        part_key = (part.get("key") or "").strip()
        part_id = parse_part_id_from_url(part_key)
        title = extract_history_title(video)

        out.append({
            "url": build_plex_part_url(part_key),
            "title": title,
            "type": item_type,
            "position": max(0.0, offset_sec),
            "known_duration": duration_sec if duration_sec > 0 else None,
            "updated_at": updated_at_ms,
            "plex_updated_at": updated_at_ms,
            "content_id": content_id,
            "part_id": part_id,
            "source": "plex",
        })
        if len(out) >= int(limit):
            break
    return out


def build_plex_part_url(part_key):
    cfg = load_plex_config()
    token = urllib.parse.quote(cfg["token"])
    if "?" in part_key:
        return f"{cfg['base_url']}{part_key}&download=1&X-Plex-Token={token}"
    return f"{cfg['base_url']}{part_key}?download=1&X-Plex-Token={token}"


def resolve_plex_item(content_id, item_type):
    def build_resolved(video_node, part_node, resolved_title, fallback_content_id):
        duration_sec = None
        start_sec = 0.0
        try:
            raw_duration = float(video_node.get("duration", "0") or "0")
            if raw_duration > 0:
                duration_sec = raw_duration / 1000.0
        except (TypeError, ValueError):
            duration_sec = None
        try:
            raw_offset = float(video_node.get("viewOffset", "0") or "0")
            if raw_offset > 0:
                start_sec = raw_offset / 1000.0
        except (TypeError, ValueError):
            start_sec = 0.0
        if start_sec <= 0:
            try:
                if int(video_node.get("viewCount", "0") or "0") > 0 and duration_sec and duration_sec > 0:
                    start_sec = duration_sec
            except (TypeError, ValueError):
                pass
        return {
            "url": build_plex_part_url(part_node.get("key")),
            "title": resolved_title,
            "content_id": str(video_node.get("ratingKey") or fallback_content_id or ""),
            "part_id": parse_part_id_from_url(part_node.get("key")),
            "start_position": max(0.0, float(start_sec or 0.0)),
            "known_duration": duration_sec if duration_sec and duration_sec > 0 else None,
        }

    item_type = (item_type or "movie").lower()
    if item_type == "episode":
        root = plex_get_xml(f"/library/metadata/{content_id}")
        video = root.find("./Video")
        if video is None:
            raise RuntimeError("No episode found")
        part = video.find(".//Part")
        if part is None or not part.get("key"):
            raise RuntimeError("No playable part found for episode")
        series_title = video.get("grandparentTitle") or "Series"
        ep_title = video.get("title") or "Episode"
        return build_resolved(video, part, f"{series_title} - {ep_title}", content_id)

    if item_type == "season":
        root = plex_get_xml(f"/library/metadata/{content_id}/children")
        video = root.find("./Video")
        if video is None:
            raise RuntimeError("No episodes found for season")
        part = video.find(".//Part")
        if part is None or not part.get("key"):
            raise RuntimeError("No playable part found for season")
        series_title = video.get("grandparentTitle") or "Series"
        ep_title = video.get("title") or "Episode"
        return build_resolved(video, part, f"{series_title} - {ep_title}", "")

    if item_type == "series":
        root = plex_get_xml(f"/library/metadata/{content_id}/allLeaves")
        video = root.find("./Video")
        if video is None:
            raise RuntimeError("No episode found for series")
        part = video.find(".//Part")
        if part is None or not part.get("key"):
            raise RuntimeError("No playable part found for selected series")
        title = video.get("grandparentTitle") or "Series"
        ep = video.get("title") or "Episode"
        return build_resolved(video, part, f"{title} - {ep}", "")

    root = plex_get_xml(f"/library/metadata/{content_id}")
    video = root.find("./Video")
    if video is None:
        raise RuntimeError("No media found for item")
    part = video.find(".//Part")
    if part is None or not part.get("key"):
        raise RuntimeError("No playable part found for item")
    title = video.get("title") or "Movie"
    return build_resolved(video, part, title, content_id)


def plex_list_seasons(show_id):
    root = plex_get_xml(f"/library/metadata/{show_id}/children")
    out = []
    for season in root.findall("./Directory"):
        out.append({
            "id": season.get("ratingKey", ""),
            "title": season.get("title", "Season"),
            "index": int(season.get("index", "0") or "0"),
            "leaf_count": int(season.get("leafCount", "0") or "0"),
        })
    out.sort(key=lambda x: x["index"])
    return out


def plex_list_episodes(season_id):
    root = plex_get_xml(f"/library/metadata/{season_id}/children")
    out = []
    for ep in root.findall("./Video"):
        out.append({
            "id": ep.get("ratingKey", ""),
            "title": ep.get("title", "Episode"),
            "index": int(ep.get("index", "0") or "0"),
            "season_index": int(ep.get("parentIndex", "0") or "0"),
        })
    out.sort(key=lambda x: x["index"])
    return out


def is_valid_http_url(value):
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def parse_start_seconds(params):
    raw = ((params.get("start") or ["0"])[0]).strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def parse_int_param(params, key, min_value, max_value):
    raw = ((params.get(key) or [""])[0]).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(min_value, min(max_value, value))


def is_likely_live_stream(video_url):
    value = video_url.lower()
    return (
        "/proxy/ts/stream/" in value
        or ".m3u8" in value
        or "/live/" in value
        or ("stream.mpg" in value and "format=ts" in value)
        or "codec=copy" in value
    )


def is_mpegts_like_stream(video_url):
    value = video_url.lower()
    return (
        "format=ts" in value
        or value.endswith(".ts")
        or value.endswith(".m2ts")
        or ("stream.mpg" in value and "codec=copy" in value)
    )


def is_local_input_path(video_url):
    return video_url.startswith("/") or video_url.startswith("./") or video_url.startswith("../")


def parse_channels_m3u(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (splay)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    channels = []
    pending_name = ""
    pending_group = ""
    pending_logo = ""
    for line in lines:
        if line.startswith("#EXTINF:"):
            pending_name = ""
            pending_group = ""
            pending_logo = ""
            attrs = line
            group_match = re.search(r'group-title="([^"]+)"', attrs, flags=re.IGNORECASE)
            if group_match:
                pending_group = group_match.group(1).strip()
            logo_match = re.search(r'tvg-logo="([^"]+)"', attrs, flags=re.IGNORECASE)
            if logo_match:
                pending_logo = logo_match.group(1).strip()
            if "," in line:
                pending_name = line.rsplit(",", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        if line.lower().startswith(("http://", "https://")):
            channels.append({
                "name": pending_name or line,
                "url": line,
                "group": pending_group or "",
                "logo": pending_logo or "",
            })
            pending_name = ""
            pending_group = ""
            pending_logo = ""
    return channels


def _normalize_plex_history_entry(entry):
    if not isinstance(entry, dict):
        return None
    url = str(entry.get("url", "")).strip()
    title = str(entry.get("title", "")).strip()
    if not url or not title:
        return None
    item_type = str(entry.get("type", "item") or "item").strip().lower()
    try:
        position = float(entry.get("position", 0) or 0)
    except (TypeError, ValueError):
        position = 0.0
    try:
        known_duration = float(entry.get("known_duration")) if entry.get("known_duration") is not None else None
    except (TypeError, ValueError):
        known_duration = None
    try:
        updated_at = int(entry.get("updated_at") or int(time.time() * 1000))
    except (TypeError, ValueError):
        updated_at = int(time.time() * 1000)
    try:
        plex_updated_at = int(entry.get("plex_updated_at")) if entry.get("plex_updated_at") is not None else None
    except (TypeError, ValueError):
        plex_updated_at = None
    content_id = str(entry.get("content_id", "") or "").strip()
    part_id = str(entry.get("part_id", "") or "").strip()
    source = str(entry.get("source", "") or "").strip().lower()
    if not content_id and part_id:
        content_id = resolve_content_id_from_part_id(part_id)
    if not part_id:
        part_id = parse_part_id_from_url(url)

    return {
        "url": url,
        "title": title,
        "type": item_type if item_type else "item",
        "position": position if position >= 0 else 0.0,
        "known_duration": known_duration if (known_duration is not None and known_duration > 0) else None,
        "updated_at": updated_at,
        "plex_updated_at": plex_updated_at if plex_updated_at and plex_updated_at > 0 else None,
        "content_id": content_id,
        "part_id": part_id,
        "source": source if source else None,
    }


def _load_plex_history_entries():
    if not PLEX_HISTORY_PATH.exists():
        return []
    try:
        raw = json.loads(PLEX_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    items = []
    for item in raw:
        normalized = _normalize_plex_history_entry(item)
        if normalized is not None:
            items.append(normalized)
    items.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    return items[:PLEX_HISTORY_MAX_ITEMS]


def _save_plex_history_entries(items):
    cleaned = []
    for item in items:
        normalized = _normalize_plex_history_entry(item)
        if normalized is not None:
            cleaned.append(normalized)
    cleaned.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    payload = cleaned[:PLEX_HISTORY_MAX_ITEMS]
    PLEX_HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def list_plex_history():
    with PLEX_HISTORY_LOCK:
        return _load_plex_history_entries()


def upsert_plex_history(entry):
    normalized = _normalize_plex_history_entry(entry)
    if normalized is None:
        return None
    with PLEX_HISTORY_LOCK:
        items = _load_plex_history_entries()
        target_content_id = str(normalized.get("content_id") or "").strip()
        target_url = str(normalized.get("url") or "").strip()
        index = -1
        if target_content_id:
            index = next((i for i, item in enumerate(items) if str(item.get("content_id") or "").strip() == target_content_id), -1)
        if index < 0 and target_url:
            index = next((i for i, item in enumerate(items) if item.get("url") == target_url), -1)
        if index >= 0:
            existing = items[index]
            merged = {**existing, **normalized}
            existing_ts = int(existing.get("updated_at") or 0)
            incoming_ts = int(normalized.get("updated_at") or 0)
            if existing_ts > incoming_ts:
                merged["updated_at"] = existing_ts
            existing_plex_ts = int(existing.get("plex_updated_at") or 0)
            incoming_plex_ts = int(normalized.get("plex_updated_at") or 0)
            if existing_plex_ts > incoming_plex_ts:
                merged["plex_updated_at"] = existing_plex_ts
            items[index] = merged
        else:
            items.append(normalized)
        saved = _save_plex_history_entries(items)
    if target_content_id:
        matched = next((item for item in saved if str(item.get("content_id") or "").strip() == target_content_id), None)
        if matched is not None:
            return matched
    return next((item for item in saved if item.get("url") == target_url), normalized)


def merge_plex_history_from_server(entries):
    imported = 0
    updated = 0
    unchanged = 0
    skipped = 0
    with PLEX_HISTORY_LOCK:
        items = _load_plex_history_entries()
        for raw in entries:
            normalized = _normalize_plex_history_entry(raw)
            if normalized is None:
                skipped += 1
                continue
            target_content_id = str(normalized.get("content_id") or "").strip()
            target_url = str(normalized.get("url") or "").strip()
            index = -1
            if target_content_id:
                index = next((i for i, item in enumerate(items) if str(item.get("content_id") or "").strip() == target_content_id), -1)
            if index < 0 and target_url:
                index = next((i for i, item in enumerate(items) if str(item.get("url") or "").strip() == target_url), -1)
            if index < 0:
                items.append(normalized)
                imported += 1
                continue

            current = items[index]
            current_ts = int(current.get("updated_at") or 0)
            incoming_ts = int(normalized.get("updated_at") or 0)
            if incoming_ts > current_ts:
                items[index] = {**current, **normalized, "source": "merged"}
                updated += 1
            elif incoming_ts == current_ts and float(normalized.get("position") or 0) > float(current.get("position") or 0):
                items[index] = {**current, **normalized, "source": "merged"}
                updated += 1
            else:
                unchanged += 1
        saved = _save_plex_history_entries(items)
    return {
        "imported": imported,
        "updated": updated,
        "unchanged": unchanged,
        "skipped": skipped,
        "total": len(saved),
    }


def delete_plex_history(url):
    target = str(url or "").strip()
    if not target:
        return False
    with PLEX_HISTORY_LOCK:
        items = _load_plex_history_entries()
        next_items = [item for item in items if item.get("url") != target]
        if len(next_items) == len(items):
            return False
        _save_plex_history_entries(next_items)
        return True


def build_input_args(video_url, start_seconds, for_audio=False):
    if is_local_input_path(video_url):
        args = []
        if start_seconds > 0:
            args.extend(["-ss", f"{start_seconds:.3f}"])
        args.extend(["-i", video_url])
        return args

    live = is_likely_live_stream(video_url)
    mpegts_like = is_mpegts_like_stream(video_url)
    args = [
        "-reconnect",
        "1",
        "-reconnect_at_eof",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "4xx,5xx",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "2",
        "-user_agent",
        "Mozilla/5.0 (splay)",
    ]
    if live:
        # Do not use -re on live network inputs: it can delay first-frame startup.
        probe = LIVE_AUDIO_PROBESIZE if for_audio else LIVE_PROBESIZE
        analyze = LIVE_AUDIO_ANALYZEDURATION if for_audio else LIVE_ANALYZEDURATION
        if mpegts_like:
            # TS-over-HTTP streams often need explicit demux hinting for valid first frames.
            args.extend([
                "-f",
                "mpegts",
            ])
        if for_audio:
            # Keep audio latency low for live sync while tolerating minor transport errors.
            args.extend([
                "-fflags",
                "+genpts+discardcorrupt+nobuffer",
                "-flags",
                "low_delay",
                "-err_detect",
                "ignore_err",
                "-probesize",
                probe,
                "-analyzeduration",
                analyze,
            ])
        else:
            # Keep video startup robust: prefer getting a decodable first frame quickly.
            args.extend([
                "-fflags",
                "+genpts+discardcorrupt",
                "-err_detect",
                "ignore_err",
                "-probesize",
                probe,
                "-analyzeduration",
                analyze,
            ])
    else:
        args.insert(0, "-re")
        if start_seconds > 0:
            args.extend(["-ss", f"{start_seconds:.3f}"])

    args.extend(["-i", video_url])
    return args


def probe_media_duration_seconds(video_url):
    if is_likely_live_stream(video_url):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-user_agent",
        "Mozilla/5.0 (splay)",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        raw = (result.stdout or "").strip().splitlines()
        if not raw:
            return None
        value = float(raw[-1].strip())
        if value > 0 and value < 60 * 60 * 24 * 10:
            return value
    except Exception:
        return None
    return None


def vaapi_decode_available():
    return Path(VAAPI_DEVICE).exists()


def cleanup_sessions():
    now = time.time()
    with SESSIONS_LOCK:
        stale = [
            sid
            for sid, data in SESSIONS.items()
            if (now - data.get("last_seen", data.get("created_at", now))) > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            session = SESSIONS.pop(sid, None)
            if session:
                stop_live_buffer_session(session)


def create_session(video_url):
    sid = uuid.uuid4().hex
    now = time.time()
    with SESSIONS_LOCK:
        SESSIONS[sid] = {
            "url": video_url,
            "play_url": video_url,
            "mode": "direct",
            "video_renderer": "mjpeg",
            "video_encoder": "cpu",
            "video_clock": 0.0,
            "video_raw_clock": None,
            "video_clock_origin": None,
            "video_clock_updated_at": now,
            "audio_clock": 0.0,
            "audio_raw_clock": None,
            "audio_clock_origin": None,
            "audio_clock_updated_at": now,
            "video_bw_samples": [],
            "audio_bw_samples": [],
            "created_at": now,
            "last_seen": now,
            "buffer_proc": None,
            "buffer_dir": None,
            "buffer_playlist": None,
        }
    return sid


def get_live_buffer_paths(sid):
    session_dir = LIVE_BUFFER_ROOT / sid
    playlist_path = session_dir / "index.m3u8"
    segment_pattern = str(session_dir / "seg_%06d.ts")
    return session_dir, playlist_path, segment_pattern


def stop_live_buffer_session(session):
    proc = session.get("buffer_proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
    session["buffer_proc"] = None
    buffer_dir = session.get("buffer_dir")
    if buffer_dir:
        try:
            shutil.rmtree(buffer_dir, ignore_errors=True)
        except Exception:
            pass
    session["buffer_dir"] = None
    session["buffer_playlist"] = None


def wait_for_live_buffer_ready(playlist_path, timeout_seconds):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if playlist_path.exists():
                text = playlist_path.read_text(encoding="utf-8", errors="ignore")
                segments = [line for line in text.splitlines() if line and not line.startswith("#")]
                if len(segments) >= LIVE_BUFFER_MIN_SEGMENTS:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_live_buffer_for_session(sid, video_url):
    session_dir, playlist_path, segment_pattern = get_live_buffer_paths(sid)
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass
    session_dir.mkdir(parents=True, exist_ok=True)

    input_args = build_input_args(video_url, 0.0, for_audio=False)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        *input_args,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "hls",
        "-hls_time",
        str(LIVE_BUFFER_HLS_TIME_SECONDS),
        "-hls_list_size",
        "0",
        "-hls_flags",
        "append_list+independent_segments+temp_file",
        "-hls_segment_filename",
        segment_pattern,
        str(playlist_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ready = wait_for_live_buffer_ready(playlist_path, LIVE_BUFFER_READY_TIMEOUT_SECONDS)
    if not ready:
        err_tail = ""
        if proc.stderr is not None:
            try:
                raw = proc.stderr.read(2048) if proc.poll() is not None else b""
                err_tail = (raw or b"").decode("utf-8", errors="ignore").strip().replace("\n", " | ")
            except Exception:
                err_tail = ""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass
        print(
            "[splay] live buffer startup failed"
            f" sid={sid} url={video_url} stderr={err_tail or '-'}"
        )
        return None, None
    return proc, playlist_path


def session_from_params(params):
    sid = ((params.get("sid") or [""])[0]).strip()
    if sid:
        with SESSIONS_LOCK:
            session = SESSIONS.get(sid)
            if session:
                session["last_seen"] = time.time()
                return sid, session.get("play_url", session["url"])
        return None, None

    video_url = unquote((params.get("url") or [""])[0]).strip()
    if not video_url or not is_valid_http_url(video_url):
        return None, None
    return None, video_url


def get_session_mode(sid):
    if not sid:
        return None
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return None
        return session.get("mode")


def set_video_clock(sid, value):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if session:
            now = time.time()
            session["video_clock"] = max(0.0, float(value))
            session["video_clock_updated_at"] = now
            session["last_seen"] = now


def set_session_video_path(sid, renderer, encoder):
    if not sid:
        return
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if session:
            session["video_renderer"] = str(renderer or "mjpeg")
            session["video_encoder"] = str(encoder or "cpu")
            session["last_seen"] = time.time()


def reset_video_clock(sid):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if session:
            now = time.time()
            session["video_clock"] = 0.0
            session["video_raw_clock"] = None
            session["video_clock_origin"] = None
            session["video_clock_updated_at"] = now
            session["last_seen"] = now


def set_video_clock_from_raw(sid, raw_value):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return
        now = time.time()
        raw = float(raw_value)
        origin = session.get("video_clock_origin")
        if origin is None:
            origin = raw
            session["video_clock_origin"] = origin
        normalized = raw - origin
        session["video_raw_clock"] = raw
        session["video_clock"] = normalized if normalized > 0 else 0.0
        session["video_clock_updated_at"] = now
        session["last_seen"] = now


def reset_audio_clock(sid):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if session:
            now = time.time()
            session["audio_clock"] = 0.0
            session["audio_raw_clock"] = None
            session["audio_clock_origin"] = None
            session["audio_clock_updated_at"] = now
            session["last_seen"] = now


def set_audio_clock_from_raw(sid, raw_value):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return
        now = time.time()
        raw = float(raw_value)
        origin = session.get("audio_clock_origin")
        if origin is None:
            origin = raw
            session["audio_clock_origin"] = origin
        normalized = raw - origin
        session["audio_raw_clock"] = raw
        session["audio_clock"] = normalized if normalized > 0 else 0.0
        session["audio_clock_updated_at"] = now
        session["last_seen"] = now


def read_session_clocks(sid):
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return None
        session["last_seen"] = time.time()
        return {
            "video_clock": session.get("video_clock", 0.0),
            "video_raw_clock": session.get("video_raw_clock"),
            "video_origin_raw": session.get("video_clock_origin"),
            "audio_clock": session.get("audio_clock", 0.0),
            "audio_raw_clock": session.get("audio_raw_clock"),
            "audio_origin_raw": session.get("audio_clock_origin"),
        }


def _prune_bw_samples(samples, now, window_seconds):
    cutoff = now - window_seconds
    while samples and samples[0][0] < cutoff:
        samples.pop(0)


def add_session_bytes(sid, stream_type, num_bytes):
    if sid is None or num_bytes <= 0:
        return
    now = time.time()
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return
        key = "video_bw_samples" if stream_type == "video" else "audio_bw_samples"
        samples = session.get(key)
        if samples is None:
            samples = []
            session[key] = samples
        _prune_bw_samples(samples, now, 8.0)
        samples.append((now, int(num_bytes)))
        session["last_seen"] = now


def read_session_metrics(sid):
    now = time.time()
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return None
        video_samples = session.get("video_bw_samples", [])
        audio_samples = session.get("audio_bw_samples", [])
        _prune_bw_samples(video_samples, now, 8.0)
        _prune_bw_samples(audio_samples, now, 8.0)
        video_bits = sum(max(0, int(item[1])) for item in video_samples) * 8.0
        audio_bits = sum(max(0, int(item[1])) for item in audio_samples) * 8.0
        window = 5.0
        session["last_seen"] = now
        return {
            "sid": sid,
            "video_bps": video_bits / window,
            "audio_bps": audio_bits / window,
            "total_bps": (video_bits + audio_bits) / window,
            "video_renderer": str(session.get("video_renderer") or "mjpeg"),
            "video_encoder": str(session.get("video_encoder") or "cpu"),
            "ts": now,
        }


def watch_progress(proc, sid, raw_clock_setter):
    if sid is None or proc.stderr is None:
        return

    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("out_time_us="):
            try:
                micros = int(line.split("=", 1)[1])
                raw_clock_setter(sid, micros / 1_000_000.0)
            except ValueError:
                pass
        elif line.startswith("out_time_ms="):
            try:
                millis = int(line.split("=", 1)[1])
                raw_clock_setter(sid, millis / 1_000_000.0)
            except ValueError:
                pass


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        cleanup_sessions()
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.serve_index()
            return

        if parsed.path == "/session/start":
            self.serve_session_start(parsed.query)
            return

        if parsed.path == "/session/clock":
            self.serve_session_clock(parsed.query)
            return

        if parsed.path == "/session/metrics":
            self.serve_session_metrics(parsed.query)
            return

        if parsed.path == "/session/events":
            self.serve_session_events(parsed.query)
            return

        if parsed.path == "/plex/search":
            self.serve_plex_search(parsed.query)
            return

        if parsed.path == "/plex/load":
            self.serve_plex_load(parsed.query)
            return

        if parsed.path == "/plex/seasons":
            self.serve_plex_seasons(parsed.query)
            return

        if parsed.path == "/plex/episodes":
            self.serve_plex_episodes(parsed.query)
            return

        if parsed.path == "/channels/list":
            self.serve_channels_list()
            return

        if parsed.path == "/channels/logo":
            self.serve_channel_logo(parsed.query)
            return

        if parsed.path == "/history/plex":
            self.serve_plex_history_list()
            return

        if parsed.path == "/history/plex/sync/status":
            self.serve_plex_sync_status()
            return

        if parsed.path == "/mjpeg":
            self.serve_mjpeg(parsed.query)
            return

        if parsed.path == "/h264":
            self.serve_h264(parsed.query)
            return

        if parsed.path == "/audio":
            self.serve_audio(parsed.query)
            return

        if self.serve_static(parsed.path):
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        cleanup_sessions()
        parsed = urlparse(self.path)

        if parsed.path == "/history/plex/upsert":
            self.serve_plex_history_upsert()
            return

        if parsed.path == "/history/plex/delete":
            self.serve_plex_history_delete()
            return

        if parsed.path == "/history/plex/sync/pull":
            self.serve_plex_history_sync_pull()
            return

        if parsed.path == "/history/plex/sync/push":
            self.serve_plex_history_sync_push()
            return

        self.send_error(404, "Not found")

    def serve_index(self):
        if not INDEX_PATH.exists():
            self.send_error(404, "index.html not found")
            return

        data = INDEX_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_static(self, path):
        clean_path = path.lstrip("/")
        if not clean_path:
            return False
        candidate = (BASE_DIR / clean_path).resolve()
        if BASE_DIR not in candidate.parents and candidate != BASE_DIR:
            return False
        if not candidate.exists() or not candidate.is_file():
            return False

        content_type = "application/octet-stream"
        suffix = candidate.suffix.lower()
        if suffix == ".png":
            content_type = "image/png"
        elif suffix in (".jpg", ".jpeg"):
            content_type = "image/jpeg"
        elif suffix == ".gif":
            content_type = "image/gif"
        elif suffix == ".webp":
            content_type = "image/webp"
        elif suffix == ".svg":
            content_type = "image/svg+xml"
        elif suffix == ".ico":
            content_type = "image/x-icon"
        elif suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif suffix == ".html":
            content_type = "text/html; charset=utf-8"

        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)
        return True

    def serve_session_start(self, query):
        params = parse_qs(query)
        video_url = unquote((params.get("url") or [""])[0]).strip()

        if not video_url:
            self.send_error(400, "Missing url parameter")
            return

        if not is_valid_http_url(video_url):
            self.send_error(400, "Only http/https URLs are allowed")
            return

        sid = create_session(video_url)
        payload = {"sid": sid, "mode": "direct"}
        live_input = is_likely_live_stream(video_url)
        if live_input:
            proc, playlist_path = start_live_buffer_for_session(sid, video_url)
            if proc is not None and playlist_path is not None:
                with SESSIONS_LOCK:
                    session = SESSIONS.get(sid)
                    if session:
                        session["buffer_proc"] = proc
                        session["buffer_dir"] = str(playlist_path.parent)
                        session["buffer_playlist"] = str(playlist_path)
                        session["play_url"] = str(playlist_path)
                        session["mode"] = "buffered_live"
                payload["mode"] = "buffered_live"
            else:
                payload["mode"] = "live_passthrough"

        duration = probe_media_duration_seconds(video_url)
        if duration is not None:
            payload["duration"] = round(duration, 3)
        self.send_json(payload)

    def serve_session_clock(self, query):
        params = parse_qs(query)
        sid = ((params.get("sid") or [""])[0]).strip()

        if not sid:
            self.send_error(400, "Missing sid parameter")
            return

        clocks = read_session_clocks(sid)
        if clocks is None:
            self.send_error(404, "Session not found")
            return

        self.send_json({"sid": sid, **clocks})

    def serve_session_metrics(self, query):
        params = parse_qs(query)
        sid = ((params.get("sid") or [""])[0]).strip()

        if not sid:
            self.send_error(400, "Missing sid parameter")
            return

        metrics = read_session_metrics(sid)
        if metrics is None:
            self.send_error(404, "Session not found")
            return

        self.send_json(metrics)

    def serve_session_events(self, query):
        params = parse_qs(query)
        sid = ((params.get("sid") or [""])[0]).strip()

        if not sid:
            self.send_error(400, "Missing sid parameter")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                clocks = read_session_clocks(sid)
                if clocks is None:
                    self.wfile.write(b"event: close\ndata: {}\n\n")
                    self.wfile.flush()
                    break

                payload = json.dumps({"sid": sid, **clocks, "ts": time.time()})
                self.wfile.write(f"event: clock\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_plex_search(self, query):
        params = parse_qs(query)
        q = ((params.get("q") or [""])[0]).strip()
        if not q:
            self.send_json({"results": []})
            return
        try:
            results = plex_search(q, limit=12)
            self.send_json({"results": results})
        except Exception as err:
            self.send_error(500, f"Plex search failed: {err}")

    def serve_plex_load(self, query):
        params = parse_qs(query)
        content_id = ((params.get("id") or [""])[0]).strip()
        item_type = ((params.get("type") or ["movie"])[0]).strip()
        if not content_id:
            self.send_error(400, "Missing id parameter")
            return
        try:
            resolved = resolve_plex_item(content_id, item_type)
            self.send_json(resolved)
        except Exception as err:
            self.send_error(500, f"Plex load failed: {err}")

    def serve_plex_seasons(self, query):
        params = parse_qs(query)
        show_id = ((params.get("id") or [""])[0]).strip()
        if not show_id:
            self.send_error(400, "Missing id parameter")
            return
        try:
            self.send_json({"results": plex_list_seasons(show_id)})
        except Exception as err:
            self.send_error(500, f"Plex seasons failed: {err}")

    def serve_plex_episodes(self, query):
        params = parse_qs(query)
        season_id = ((params.get("id") or [""])[0]).strip()
        if not season_id:
            self.send_error(400, "Missing id parameter")
            return
        try:
            self.send_json({"results": plex_list_episodes(season_id)})
        except Exception as err:
            self.send_error(500, f"Plex episodes failed: {err}")

    def serve_channels_list(self):
        try:
            channels = parse_channels_m3u(CHANNELS_M3U_URL)
            self.send_json({"results": channels})
        except Exception as err:
            self.send_error(500, f"Channels load failed: {err}")

    def serve_channel_logo(self, query):
        params = parse_qs(query)
        logo_url = unquote((params.get("url") or [""])[0]).strip()
        if not logo_url or not is_valid_http_url(logo_url):
            self.send_error(400, "Missing/invalid logo url")
            return
        req = urllib.request.Request(logo_url, headers={"User-Agent": "Mozilla/5.0 (splay)"})
        try:
            with urllib.request.urlopen(req, timeout=12) as res:
                content_type = (res.headers.get("Content-Type") or "").strip()
                data = res.read(2 * 1024 * 1024 + 1)
        except Exception as err:
            self.send_error(502, f"Logo fetch failed: {err}")
            return
        if len(data) > 2 * 1024 * 1024:
            self.send_error(413, "Logo too large")
            return
        if not content_type:
            guessed, _ = mimetypes.guess_type(logo_url)
            content_type = guessed or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=21600")
        self.end_headers()
        self.wfile.write(data)

    def serve_plex_history_list(self):
        try:
            self.send_json({"results": list_plex_history()})
        except Exception as err:
            self.send_error(500, f"Plex history load failed: {err}")

    def serve_plex_history_upsert(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_error(400, "Missing request body")
            return
        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return
        item = upsert_plex_history(payload)
        if item is None:
            self.send_error(400, "Invalid history payload")
            return
        self.send_json({"ok": True, "item": item})

    def serve_plex_history_delete(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_error(400, "Missing request body")
            return
        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return
        url = ""
        if isinstance(payload, dict):
            url = str(payload.get("url", "")).strip()
        if not url:
            self.send_error(400, "Missing url")
            return
        removed = delete_plex_history(url)
        self.send_json({"ok": True, "removed": bool(removed), "url": url})

    def serve_plex_sync_status(self):
        with PLEX_SYNC_LOCK:
            self.send_json({"ok": True, **PLEX_SYNC_STATUS})

    def serve_plex_history_sync_pull(self):
        try:
            incoming = plex_fetch_recent_history(limit=PLEX_PULL_MAX_ITEMS)
            merged = merge_plex_history_from_server(incoming)
            with PLEX_SYNC_LOCK:
                PLEX_SYNC_STATUS["last_pull_at"] = int(time.time() * 1000)
                PLEX_SYNC_STATUS["last_pull_error"] = ""
            self.send_json({"ok": True, **merged})
        except Exception as err:
            with PLEX_SYNC_LOCK:
                PLEX_SYNC_STATUS["last_pull_error"] = str(err)
            self.send_error(500, f"Plex history sync pull failed: {err}")

    def serve_plex_history_sync_push(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_error(400, "Missing request body")
            return
        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(400, "Invalid payload")
            return

        content_id = str(payload.get("content_id", "") or "").strip()
        if not content_id:
            url = str(payload.get("url", "") or "").strip()
            part_id = str(payload.get("part_id", "") or "").strip() or parse_part_id_from_url(url)
            if part_id:
                content_id = resolve_content_id_from_part_id(part_id)
        if not content_id:
            content_id = resolve_content_id_from_title(
                str(payload.get("title", "") or "").strip(),
                str(payload.get("type", "") or "").strip(),
            )
        if not content_id:
            self.send_error(400, "Missing content_id")
            return

        try:
            position = max(0.0, float(payload.get("position", 0) or 0))
        except (TypeError, ValueError):
            position = 0.0
        try:
            known_duration = float(payload.get("known_duration")) if payload.get("known_duration") is not None else 0.0
        except (TypeError, ValueError):
            known_duration = 0.0
        state = str(payload.get("state", "playing") or "playing").strip().lower()
        event = str(payload.get("event", "") or "").strip().lower()
        should_mark_watched = bool(payload.get("mark_watched", False))

        try:
            plex_sync_write_timeline(content_id, position, known_duration, state)
            marked = False
            if should_mark_watched:
                plex_sync_mark_watched(content_id)
                marked = True
            with PLEX_SYNC_LOCK:
                PLEX_SYNC_STATUS["last_push_at"] = int(time.time() * 1000)
                PLEX_SYNC_STATUS["last_push_error"] = ""
            self.send_json({
                "ok": True,
                "content_id": content_id,
                "position": position,
                "state": state,
                "event": event,
                "marked_watched": marked,
            })
        except Exception as err:
            with PLEX_SYNC_LOCK:
                PLEX_SYNC_STATUS["last_push_error"] = str(err)
            self.send_error(500, f"Plex history sync push failed: {err}")

    def serve_mjpeg(self, query):
        params = parse_qs(query)
        sid, video_url = session_from_params(params)
        session_mode = get_session_mode(sid)
        start_seconds = parse_start_seconds(params)
        live_input = is_likely_live_stream(video_url or "")
        fps_mode = ((params.get("fps_mode") or ["fixed"])[0]).strip().lower()
        requested_fps = parse_int_param(params, "fps", 1, 60)
        if fps_mode == "match":
            fps_value = None
        else:
            fps_value = requested_fps if requested_fps is not None else (LIVE_VIDEO_FPS if live_input else None)
        requested_q = parse_int_param(params, "q", 2, 31)
        q_value = str(requested_q if requested_q is not None else int(VIDEO_Q_CPU))

        if not video_url:
            self.send_error(400, "Missing/invalid sid or url parameter")
            return

        if sid:
            reset_video_clock(sid)

        input_args = build_input_args(video_url, start_seconds, for_audio=False)
        if session_mode == "buffered_live" and "-re" not in input_args:
            # Pace buffered-live playback at wall clock to avoid bursty MJPEG output.
            input_args = ["-re", *input_args]
        video_filter = f"scale='min({VIDEO_WIDTH},iw)':-1:flags=lanczos"
        if fps_value is not None:
            video_filter = f"fps={fps_value}," + video_filter

        cpu_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:2",
            *input_args,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            video_filter,
            "-q:v",
            q_value,
            "-f",
            "mpjpeg",
            "pipe:1",
        ]

        hwdecode_cmd = None
        # For live sources, prefer a deterministic CPU path for first-frame reliability.
        if (not live_input) and vaapi_decode_available():
            hwdecode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-progress",
                "pipe:2",
                "-hwaccel",
                "vaapi",
                "-hwaccel_device",
                VAAPI_DEVICE,
                "-hwaccel_output_format",
                "vaapi",
                *input_args,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                ("fps=" + str(fps_value) + "," if fps_value is not None else "") + f"hwdownload,format=nv12,scale='min({VIDEO_WIDTH},iw)':-1:flags=lanczos",
                "-q:v",
                q_value,
                "-f",
                "mpjpeg",
                "pipe:1",
            ]

        base_env = dict(os.environ)
        # Respect host VAAPI driver selection; avoid forcing i965 universally.
        proc = subprocess.Popen(hwdecode_cmd or cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=base_env)
        first_chunk_timeout = LIVE_FIRST_CHUNK_TIMEOUT_SECONDS if live_input else DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS
        first_chunk = self.read_initial_chunk(proc, timeout_seconds=first_chunk_timeout)

        if not first_chunk and hwdecode_cmd is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc = subprocess.Popen(cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=base_env)
            first_chunk = self.read_initial_chunk(proc, timeout_seconds=first_chunk_timeout)

        if not first_chunk:
            err_tail = self.read_stderr_tail(proc)
            print(
                "[splay] mjpeg startup no first chunk"
                f" live={live_input} sid={sid or '-'}"
                f" timeout={first_chunk_timeout}s"
                f" url={video_url}"
                f" stderr={err_tail or '-'}"
            )
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.send_error(502, "Could not read first video frame from source")
            return
        if sid:
            set_session_video_path(sid, "mjpeg", "cpu")
            threading.Thread(target=watch_progress, args=(proc, sid, set_video_clock_from_raw), daemon=True).start()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=ffmpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            assert proc.stdout is not None
            if first_chunk:
                self.wfile.write(first_chunk)
                self.wfile.flush()
                add_session_bytes(sid, "video", len(first_chunk))
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                add_session_bytes(sid, "video", len(chunk))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def serve_audio(self, query):
        params = parse_qs(query)
        sid, video_url = session_from_params(params)
        session_mode = get_session_mode(sid)
        start_seconds = parse_start_seconds(params)

        if not video_url:
            self.send_error(400, "Missing/invalid sid or url parameter")
            return

        input_args = build_input_args(video_url, start_seconds, for_audio=True)
        # Do not force -re for buffered-live audio; keeping some ahead-buffer
        # makes corrective seeks effective when video gets ahead.

        audio_filter = "aresample=async=250:min_hard_comp=0.100"
        if session_mode == "buffered_live":
            # Minimize extra audio pipeline latency in buffered-live mode.
            audio_filter = "anull"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:2",
            *input_args,
            "-map",
            "0:a:0?",
            "-vn",
            "-af",
            audio_filter,
            "-c:a",
            "libmp3lame",
            "-b:a",
            "80k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "mp3",
            "pipe:1",
        ]

        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        if sid:
            reset_audio_clock(sid)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if sid:
            threading.Thread(target=watch_progress, args=(proc, sid, set_audio_clock_from_raw), daemon=True).start()

        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                add_session_bytes(sid, "audio", len(chunk))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def serve_h264(self, query):
        params = parse_qs(query)
        sid, video_url = session_from_params(params)
        session_mode = get_session_mode(sid)
        start_seconds = parse_start_seconds(params)
        live_input = is_likely_live_stream(video_url or "")
        # Keep WebCodecs H.264 playback stable across mixed source frame rates.
        # Always normalize output to 30fps for this path.
        fps_value = H264_TARGET_FPS
        requested_h264_q = parse_int_param(params, "h264_q", 10, 51)
        h264_q = str(requested_h264_q if requested_h264_q is not None else int(H264_VIDEO_CRF))

        if not video_url:
            self.send_error(400, "Missing/invalid sid or url parameter")
            return

        if sid:
            reset_video_clock(sid)

        input_args = build_input_args(video_url, start_seconds, for_audio=False)
        if session_mode == "buffered_live" and "-re" not in input_args:
            # Pace buffered-live playback at wall clock for stable A/V sync behavior.
            input_args = ["-re", *input_args]

        video_filter = f"scale='min({VIDEO_WIDTH},iw)':-2:flags=lanczos,format=yuv420p"
        if fps_value is not None:
            video_filter = f"fps={fps_value}," + video_filter

        cpu_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:2",
            *input_args,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            H264_VIDEO_PRESET,
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            h264_q,
            "-g",
            H264_GOP,
            "-keyint_min",
            H264_GOP,
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-f",
            "h264",
            "pipe:1",
        ]

        vaapi_filter = f"scale='min({VIDEO_WIDTH},iw)':-2:flags=lanczos,format=nv12,hwupload"
        if fps_value is not None:
            vaapi_filter = f"fps={fps_value}," + vaapi_filter

        vaapi_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:2",
            "-init_hw_device",
            f"vaapi=va:{VAAPI_DEVICE}",
            "-filter_hw_device",
            "va",
            *input_args,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            vaapi_filter,
            "-c:v",
            "h264_vaapi",
            "-qp",
            h264_q,
            "-g",
            H264_GOP,
            "-bf",
            "0",
            "-f",
            "h264",
            "pipe:1",
        ]

        encoder_used = "vaapi"
        proc = subprocess.Popen(vaapi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ))
        first_chunk_timeout = LIVE_FIRST_CHUNK_TIMEOUT_SECONDS if live_input else DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS
        first_chunk = self.read_initial_chunk(proc, timeout_seconds=first_chunk_timeout)

        if not first_chunk:
            vaapi_err_tail = self.read_stderr_tail(proc)
            print(
                "[splay] h264 vaapi startup failed; falling back to libx264"
                f" live={live_input} sid={sid or '-'}"
                f" url={video_url}"
                f" stderr={vaapi_err_tail or '-'}"
            )
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc = subprocess.Popen(cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ))
            encoder_used = "cpu"
            first_chunk = self.read_initial_chunk(proc, timeout_seconds=first_chunk_timeout)

        if not first_chunk:
            err_tail = self.read_stderr_tail(proc)
            print(
                "[splay] h264 startup no first chunk"
                f" live={live_input} sid={sid or '-'}"
                f" timeout={first_chunk_timeout}s"
                f" url={video_url}"
                f" stderr={err_tail or '-'}"
            )
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.send_error(502, "Could not read first H.264 chunk from source")
            return
        if sid:
            set_session_video_path(sid, "h264", encoder_used)
            threading.Thread(target=watch_progress, args=(proc, sid, set_video_clock_from_raw), daemon=True).start()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "video/h264")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            assert proc.stdout is not None
            self.wfile.write(first_chunk)
            self.wfile.flush()
            add_session_bytes(sid, "video", len(first_chunk))
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                add_session_bytes(sid, "video", len(chunk))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_initial_chunk(self, proc, timeout_seconds):
        if proc.stdout is None:
            return b""
        ready, _, _ = select.select([proc.stdout], [], [], timeout_seconds)
        if not ready:
            return b""
        return proc.stdout.read(8192) or b""

    def read_stderr_tail(self, proc, max_bytes=2048):
        if proc.stderr is None:
            return ""
        chunks = []
        total = 0
        try:
            while True:
                ready, _, _ = select.select([proc.stderr], [], [], 0)
                if not ready:
                    break
                data = os.read(proc.stderr.fileno(), 512)
                if not data:
                    break
                chunks.append(data)
                total += len(data)
                if total >= max_bytes:
                    break
        except Exception:
            return ""
        if not chunks:
            return ""
        raw = b"".join(chunks)[-max_bytes:]
        return raw.decode("utf-8", errors="ignore").strip().replace("\n", " | ")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from difflib import SequenceMatcher
import json
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
LIVE_FIRST_CHUNK_TIMEOUT_SECONDS = 25.0
DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 4.0
LIVE_BUFFER_ROOT = Path("/tmp/splay_live_buffer")
LIVE_BUFFER_READY_TIMEOUT_SECONDS = 20.0
LIVE_BUFFER_MIN_SEGMENTS = 3
LIVE_BUFFER_HLS_TIME_SECONDS = 2
PLEX_CONTROL_DIR = Path("/mnt/synology/misc/dev/plex_control")
PLEX_CTL_PATH = PLEX_CONTROL_DIR / "plexctl.py"
PLEX_MAPPING_PATH = PLEX_CONTROL_DIR / "plex_content_mapping.json"

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
PLEX_CFG_LOCK = threading.Lock()
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
        results.append({
            "title": title or "Untitled",
            "type": item_type,
            "content_id": content_id,
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


def plex_get_xml(path):
    cfg = load_plex_config()
    sep = "&" if "?" in path else "?"
    url = f"{cfg['base_url']}{path}{sep}X-Plex-Token={urllib.parse.quote(cfg['token'])}"
    req = urllib.request.Request(url, headers={"User-Agent": "splay/1.0"})
    with urllib.request.urlopen(req, timeout=20) as res:
        data = res.read()
    return ET.fromstring(data)


def build_plex_part_url(part_key):
    cfg = load_plex_config()
    token = urllib.parse.quote(cfg["token"])
    if "?" in part_key:
        return f"{cfg['base_url']}{part_key}&download=1&X-Plex-Token={token}"
    return f"{cfg['base_url']}{part_key}?download=1&X-Plex-Token={token}"


def resolve_plex_item(content_id, item_type):
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
        return {"url": build_plex_part_url(part.get("key")), "title": f"{series_title} - {ep_title}"}

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
        return {"url": build_plex_part_url(part.get("key")), "title": f"{series_title} - {ep_title}"}

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
        return {"url": build_plex_part_url(part.get("key")), "title": f"{title} - {ep}"}

    root = plex_get_xml(f"/library/metadata/{content_id}")
    video = root.find("./Video")
    if video is None:
        raise RuntimeError("No media found for item")
    part = video.find(".//Part")
    if part is None or not part.get("key"):
        raise RuntimeError("No playable part found for item")
    title = video.get("title") or "Movie"
    return {"url": build_plex_part_url(part.get("key")), "title": title}


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

        if parsed.path == "/mjpeg":
            self.serve_mjpeg(parsed.query)
            return

        if parsed.path == "/audio":
            self.serve_audio(parsed.query)
            return

        if self.serve_static(parsed.path):
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
        elif sid:
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

# Splay

Low-latency web video player with server-side transcoding and sync logic.

It serves a single-page UI (`index.html`) from a Python HTTP server (`server.py`), and uses `ffmpeg` for MJPEG/H.264 video + audio streaming.

## Features

- URL playback for remote media sources
- Two video renderers:
  - MJPEG (legacy)
  - WebCodecs H.264
- Fixed 30fps output pipeline
- Overlay transport controls (play/pause, fullscreen, mute)
- Touch-friendly seek/timeline controls
- Channel picker endpoint + modal UI
- Plex search/history + sync endpoints (optional integration)
- Real-time sync, FPS, and bandwidth monitoring

## Requirements

- Python 3.9+
- `ffmpeg` available on PATH
- Modern browser (Chrome/Edge recommended for WebCodecs path)

Optional (for Plex features):

- Plex control files expected by `server.py`:
  - `PLEX_CONTROL_DIR` default: `/mnt/synology/misc/dev/plex_control`
  - `plexctl.py` and `plex_content_mapping.json`

## Run

From repo root:

```bash
python3 server.py
```

Server defaults:

- Host: `0.0.0.0`
- Port: `8000`

Open:

```text
http://localhost:8000
```

## Key Files

- `server.py`: API + stream endpoints, session/sync logic, ffmpeg process orchestration
- `index.html`: player UI, controls, overlay behavior, WebCodecs/MJPEG client logic
- `plex_history.json`: local persisted Plex history cache

## Notes

- Hardware acceleration path targets VAAPI (`/dev/dri/renderD128`) when available.
- If Plex paths are missing, core playback still works; Plex-specific actions may fail.
- Some repo files are local/runtime artifacts (for example screenshots and history JSON).


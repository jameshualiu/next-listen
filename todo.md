# next-listen — Core ML Pipeline

Branch: `feature/core-pipeline` (already created, off `main` @ `3a71e9a`).

Full design rationale lives in the approved Claude Code plan (not a repo
file — session-local), but everything load-bearing is captured below so
nothing depends on that file surviving.

## Context / key decisions

- Spotify killed live playlist-discovery for apps created after Nov 27,
  2024 (Featured/Category Playlists, Related Artists gone; as of Feb/Mar
  2026, `GET /playlists/{id}/items` only returns track contents for
  playlists you own/collaborate on). **No live path to a public playlist
  corpus.** → Signal 1 trains on the Kaggle mirror of Spotify's own
  Million Playlist Dataset (https://www.kaggle.com/datasets/himanshuwagh/spotify-million),
  downloaded manually into `data/raw/mpd/` (not scripted), subset of a
  few thousand playlists. This is now documented in `CLAUDE.md`'s Known
  Hard Constraints.
- This branch builds **only** the 6 scripts CLAUDE.md's Repository Layout
  names, plus one shared `config.py`, operating on local files
  (`data/raw/`, `data/processed/`, `results/`). No pgvector persistence,
  no FastAPI `main.py` — deferred to a separate follow-up branch even
  though docker-compose/requirements.txt already provision them.
- `evaluate.py` runs in two modes: real (your actual data) or fixture
  (committed fake data under `tests/fixtures/`) so `.github/workflows/evaluate.yml`
  stays green in CI with no Spotify/Anthropic credentials.

## Implementation status

- [x] `src/config.py` — paths, env loading, logger, `require_env()`
- [x] `src/auth.py` — Spotify Authorization Code Flow via spotipy, `.spotify_cache`
- [x] `src/fetch.py` — top tracks/artists (per time range), saved tracks,
      recently played, artist genres (one-at-a-time fallback), MPD slice
      locator (`load_mpd_slice_paths`/`iter_mpd_slices`)
- [x] `src/train_embeddings.py` — Word2Vec on MPD subset, `track_catalog.json`,
      cold-start contract (`is_in_vocabulary`)
- [x] `src/genre_similarity.py` — TF-IDF (genre-as-token) + cosine similarity
- [x] `src/recommend.py` — candidate generation, signal blending + "why",
      NL query layer (Claude Haiku, forced tool use), CLI w/ clarification
      + confirmation loop
- [x] `src/evaluate.py` — 80/20 holdout, precision@10/recall@10, weight
      sweep, random baseline, real/fixture mode switch, writes `results/`
- [x] `tests/fixtures/` — `mpd_playlists.json`, `artist_genres.json`,
      `liked_tracks.json` (tiny hand-built dataset for fixture mode)
- [x] Verified: all modules import/compile cleanly; `python src/evaluate.py`
      runs fixture mode end-to-end (precision/recall computed, report
      written, exit 0) — this is exactly what CI will run
- [x] `CLAUDE.md` updated with the MPD-corpus constraint

## Not done yet — needs you, not more code

- [ ] Register `http://127.0.0.1:8080/callback` as a redirect URI in the
      Spotify Developer Dashboard (if not already done)
- [ ] Fill in real `.env` (`cp .env.example .env`) with Spotify +
      Gemini credentials
- [ ] Run `python src/auth.py` once to do the browser OAuth consent
- [ ] Manually download the Kaggle MPD mirror and extract
      `mpd.slice.*.json` files into `data/raw/mpd/`
- [ ] Run `python src/fetch.py` (real Spotify data)
- [ ] Run `python src/train_embeddings.py` (real training — this is the
      "don't run this at night" step)
- [ ] Run `python src/genre_similarity.py`
- [ ] Try `python src/recommend.py --no-llm` to sanity-check scoring
      before wiring up real NL queries
- [ ] Run `python src/evaluate.py` in real mode, review `results/latest.md`

## Known open items (flagged, not blocking)

- MPD field names/slice-file naming are based on the well-documented
  original schema, not byte-verified against the specific Kaggle mirror
  copy — the loader is defensive (recursive glob) but confirm once
  downloaded.
- Assumes `/me/top/tracks`, `/me/top/artists`, `/me/tracks`,
  `/me/player/recently-played`, `/artists/{id}`, and `/search` all still
  work in Development Mode post-Nov-2024 — not exhaustively re-verified
  beyond what surfaced in research.

## Review / commit

- [ ] You review the 7 new `src/*.py` files + `tests/fixtures/*` (all
      currently untracked, nothing staged or committed yet)
- [ ] Stage + commit (conventional-commit style per `AGENTS.md`) once approved
- [ ] Push branch / open PR — only when you explicitly say so

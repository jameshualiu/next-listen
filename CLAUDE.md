# CLAUDE.md — next-listen Context & Guidelines

## Project Intent & Working Style
* **Purpose:** Personal ML/DS platform recommending songs via Spotify history + NL query interface. 
* **Design Philosophy:** Prioritize **explainable, defensible machine learning architectures** over black-box complexity. Every recommendation must output a human-readable, deterministic audit trail explaining "why" it was selected.
* **Execution Rules:** Strictly follow the multi-phase workflow defined in `agents.md` for branch creation, staging, and Pull Requests.
* **Maintenance:** Update this file immediately whenever an architectural decision or directory structure changes.

## Critical Operational Commands
* **Run Evaluation:** `python src/evaluate.py` (Must run as a single script producing a clear metrics report)
* **Start Database:** `docker-compose up -d` (Spins up local Postgres + pgvector)
* **Run Dev Backend:** `fastapi dev src/main.py`
* **Run Frontend:** `npm run dev` (inside nextjs directory)

## MVP Scope Boundaries (Follow Strictly)
### Build This:
* **Data Fetching:** Spotify Web API (OAuth, top tracks/artists, saved tracks, recently played).
* **Signal 1 (Embeddings):** Word2Vec-style song embeddings trained on a public playlist corpus using `gensim`. (1 playlist = 1 sentence of track IDs).
* **Signal 2 (Genres):** TF-IDF + cosine similarity (`scikit-learn`) on artist genre metadata.
* **Blending:** Tunable weighted sum of normalized Signal 1 + Signal 2 scores.
* **Query Layer:** LLM intent extraction (genre + time window only). If ambiguous, ask clarifying questions instead of guessing.
* **UX Confirmation:** Display parsed intent (e.g., "genre detected / mood not detected") for confirmation before processing.
* **Evaluation:** Holdout test script (hide 20% of liked items, report precision@10 / recall@10 against a random baseline).

### DO NOT Build (Out of Scope):
* **No** audio-features APIs or third-party audio analysis.
* **No** mood-based text filtering (lacks underlying data).
* **No** long-running historical data storage beyond Spotify's 50-track cap.
* **No** cloud deployments (Vercel/Render). Keep it local-only.

## Known Hard Constraints
* **Spotify Deprecations:** Audio-features and audio-analysis endpoints return 403 errors for all apps created after Nov 27, 2024. **Do not design around or attempt to call them.**
* **No Live Playlist Corpus:** As of the same Nov 2024 change (and further restricted Feb/March 2026), Featured Playlists, Category Playlists, and Related Artists are removed for new apps, and `GET /playlists/{id}/items` only returns track contents for playlists the authenticated user owns or collaborates on. There is no live path to a public playlist corpus. **Signal 1 trains on the Kaggle mirror of Spotify's own Million Playlist Dataset** (https://www.kaggle.com/datasets/himanshuwagh/spotify-million), downloaded manually into `data/raw/mpd/` (not scripted), on a configurable subset (default 5,000 playlists).
* **History Cap:** `recently_played` is hard-capped at 50 tracks with no pagination. Lean on top-tracks aggregation for longer-term windows.
* **Cold-Start Risk:** Songs missing from the training corpus will fail embedding lookups. Implement an explicit fallback path (e.g., fallback to genre-only similarity) rather than crashing.

## Code & Repository Conventions
* **Language Split:** Python for all ML/backend (`src/`), Next.js for frontend.
* **Caching Rule:** Cache raw Spotify API responses as JSON under `data/raw/`. Reruns must read cache to avoid rate limits.
* **Notebooks Boundary:** `notebooks/` is strictly for ad-hoc experimentation. **Never** import pipeline logic from notebooks; real production logic belongs entirely in `src/`.

## Repository Layout
```text
next-listen/
├── data/{raw,processed}/
├── src/
│   ├── config.py            # Shared paths, env loading, logger
│   ├── auth.py              # Spotify OAuth
│   ├── fetch.py             # Pulls listening data + locates the MPD corpus
│   ├── train_embeddings.py  # Word2Vec training on co-occurrence
│   ├── genre_similarity.py  # TF-IDF + cosine
│   ├── recommend.py         # Blends signals, generates ranked output with "why"
│   └── evaluate.py          # Holdout precision@k / recall@k vs baseline
├── tests/fixtures/          # Tiny fake dataset so evaluate.py's CI run is credential-free
├── notebooks/explore.ipynb  # Hyperparameter tuning/exploration only
├── results/                 # Evaluation output reports
├── README.md
├── requirements.txt
└── docker-compose.yml       # Local Postgres + pgvector
```
# CLAUDE.md — next-listen Context & Guidelines

## Project Intent & Working Style
* **Purpose:** Personal ML/DS portfolio project recommending songs via Spotify history + NL query interface. 
* **Design Philosophy:** Built for job interviews. Prefer **explainable, defensible choices** over black-box complexity. Every recommendation must output a human-readable "why".
* **AI Protocol:** Start nontrivial tasks in **Plan Mode** before writing code. If something breaks, halt and re-plan.
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
│   ├── auth.py              # Spotify OAuth
│   ├── fetch.py             # Pulls listening data + playlist corpus
│   ├── train_embeddings.py  # Word2Vec training on co-occurrence
│   ├── genre_similarity.py  # TF-IDF + cosine
│   ├── recommend.py         # Blends signals, generates ranked output with "why"
│   └── evaluate.py          # Holdout precision@k / recall@k vs baseline
├── notebooks/explore.ipynb  # Hyperparameter tuning/exploration only
├── results/                 # Evaluation output reports
├── README.md
├── requirements.txt
└── docker-compose.yml       # Local Postgres + pgvector
```
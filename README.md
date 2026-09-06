# next-listen

A personal song recommendation engine built on Spotify listening history, blending Word2Vec-style co-occurrence embeddings with genre similarity, plus a natural-language query layer for filtering results.

## Why this exists

_TODO: fill in personally._

## How it works

next-listen pulls your Spotify listening history (top tracks/artists, saved tracks, recently played) via the Spotify Web API, then blends two signals to generate recommendations:

- **Signal 1 — Embeddings:** Word2Vec-style song embeddings trained on a public playlist corpus with `gensim`, treating each playlist as a "sentence" of track IDs.
- **Signal 2 — Genres:** TF-IDF + cosine similarity (`scikit-learn`) over artist genre metadata.

The two signals are combined via a tunable weighted sum, and every recommendation includes a human-readable explanation of why it was chosen. A natural-language query layer extracts intent (genre + time window only) and asks clarifying questions when the request is ambiguous, rather than guessing.

## Known limitations

- **No audio-features data:** Spotify's audio-features and audio-analysis endpoints return 403 for all apps created after November 27, 2024, so recommendations rely on embeddings and genre metadata only — no tempo/energy/valence-style filtering.
- **50-track history cap:** The `recently_played` endpoint is hard-capped at 50 tracks with no pagination, so longer-term listening patterns are approximated via top-tracks aggregation instead.

## Evaluation

_TODO: results go here once `src/evaluate.py` has been run — precision@10 / recall@10 against a random baseline on a 20% holdout of liked items._

## Setup

1. Copy the environment template and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
2. Start the local Postgres + pgvector database:
   ```bash
   docker compose up -d
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline, in order (once these scripts exist):
   ```bash
   python src/fetch.py
   python src/train_embeddings.py
   python src/evaluate.py
   ```

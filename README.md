# next-listen

A personal song recommendation engine built on Spotify listening history, blending Word2Vec-style co-occurrence embeddings with genre similarity, plus a natural-language query layer for filtering results.

## Why this exists

This project explores how to build a highly interpretable, dual-signal recommendation pipeline under strict real-world API constraints. 

Following Spotify’s late-2024 deprecation of its audio-features endpoints (tempo, energy, valence), traditional metadata-reliant recommendation approaches became unviable for new applications. **next-listen** bypasses this limitation by treating playlist curation as a natural language co-occurrence problem (using Word2Vec embeddings) and combining it with deterministic content filtering (TF-IDF on genre metadata). 

The core goal is to deliver a music discovery interface that avoids the "black box" problem of modern recommendation engines, ensuring every output can be fully audited with a human-readable explanation.

## How it works

next-listen pulls your Spotify listening history (top tracks/artists, saved tracks, recently played) via the Spotify Web API, then blends two signals to generate recommendations:

- **Signal 1 — Embeddings:** Word2Vec-style song embeddings trained on a public playlist corpus with `gensim`, treating each playlist as a "sentence" of track IDs.
- **Signal 2 — Genres:** TF-IDF + cosine similarity (`scikit-learn`) over artist genre metadata.

The two signals are combined via a tunable weighted sum, and every recommendation includes a human-readable explanation of why it was chosen. A natural-language query layer extracts intent (genre + time window only) and asks clarifying questions when the request is ambiguous, rather than guessing.

## Known limitations

- **No audio-features data:** Spotify's audio-features and audio-analysis endpoints return 403 for all apps created after November 27, 2024, so recommendations rely on embeddings and genre metadata only — no tempo/energy/valence-style filtering.
- **50-track history cap:** The `recently_played` endpoint is hard-capped at 50 tracks with no pagination, so longer-term listening patterns are approximated via top-tracks aggregation instead.

## Evaluation

*Benchmarks will be populated here dynamically once the pipeline execution report runs.* 

The evaluation script utilizes a 20% holdout split of a user's liked items to measure and report **precision@10** and **recall@10** metrics against a random baseline model.

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
"""Spotify OAuth (Authorization Code Flow) for next-listen.

Every other script that talks to Spotify goes through `get_spotify_client()`.
Run this file directly the first time to do the one-time browser consent:

    python src/auth.py

Precondition: the redirect URI in .env (default
http://127.0.0.1:8080/callback) must already be registered for this app in
the Spotify Developer Dashboard.
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth

import config

logger = config.get_logger(__name__)

CACHE_PATH = config.PROJECT_ROOT / ".spotify_cache"


def get_spotify_client(scope: str = config.SPOTIFY_SCOPES) -> spotipy.Spotify:
    """Builds an authenticated client, using a local token cache to avoid
    re-prompting for consent on every run."""
    auth_manager = SpotifyOAuth(
        client_id=config.require_env("SPOTIFY_CLIENT_ID"),
        client_secret=config.require_env("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=config.require_env("SPOTIFY_REDIRECT_URI"),
        scope=scope,
        cache_path=str(CACHE_PATH),
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


if __name__ == "__main__":
    client = get_spotify_client()
    me = client.current_user()
    logger.info("Authenticated as %s (id=%s)", me.get("display_name"), me.get("id"))

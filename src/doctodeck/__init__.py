"""doc-to-deck: upload a document, get a measured, cost-tracked flashcard deck."""

from pathlib import Path


def main() -> None:
    """Entry point for `uv run doctodeck`: starts the API and page on http://localhost:8000.

    Keys come from the environment or a git-ignored `.env` in the current folder (uvicorn loads it with
    python-dotenv, which uvicorn[standard] already installs). Tests never load it, so they stay keyless.
    """
    import uvicorn

    env_file = ".env" if Path(".env").is_file() else None
    uvicorn.run("doctodeck.app:app", host="127.0.0.1", port=8000, reload=False, env_file=env_file)

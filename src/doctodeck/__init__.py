"""doc-to-deck: upload a document, get a measured, cost-tracked flashcard deck."""


def main() -> None:
    """Entry point for `uv run doctodeck`: starts the API and page on http://localhost:8000."""
    import uvicorn

    uvicorn.run("doctodeck.app:app", host="127.0.0.1", port=8000, reload=False)

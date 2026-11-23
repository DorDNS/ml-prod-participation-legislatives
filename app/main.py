"""Minimal FastAPI application skeleton (health endpoint only)."""

from fastapi import FastAPI

app = FastAPI(title="Polling station turnout", version="0.0.1")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok"}

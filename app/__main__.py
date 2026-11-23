"""Allow `python -m app` to run the service."""

from app.main import app  # noqa: F401  (re-exported for uvicorn / -m use)

if __name__ == "__main__":
    import uvicorn

    settings = app.state.settings
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)

import os
from src.server import app


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    reload_enabled = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()

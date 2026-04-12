import os
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_fastapi_app

from .code_assist_env import CodeAssistEnv
from .models import CodeAction, CodeObservation

_root = Path(__file__).parent
_repo_root = _root.parent
_static = _repo_root / "static"
_openenv_yaml = _repo_root / "openenv.yaml"

app = create_fastapi_app(
    CodeAssistEnv,
    CodeAction,
    CodeObservation
)

class IDECpntext(BaseModel):
    content: str
    cursor_offset: int | None = None
    
class WorkspaceSync(BaseModel):
    content: str
    
@app.get("/", include_in_schema=False)
async def ide_root():
    return FileResponse(_static / "index.html")

@app.post("/workspace/sync")
async def workspace_sync(data: WorkspaceSync):
    from inference import sync_workspace
    return sync_workspace(data.content) 


@app.post("/predict")
async def predict(data: IDECpntext):
    # Triggers the RL Agent in inference.py
    from inference import get_completion
    
    return get_completion(data.content, data.cursor_offset)


def _tasks_payload() -> dict:
    from .code_assist_env import graded_tasks_manifest, graders_registry

    tasks = graded_tasks_manifest()
    graders = graders_registry()
    return {
        "tasks": tasks,
        "graders": graders,
        "task_count": len(tasks),
        "grader_count": len(graders),
        "tasks_with_graders": sum(
            1
            for t in tasks
            if t.get("grader") or (isinstance(t.get("graders"), list) and len(t["graders"]) > 0)
        ),
    }


@app.get("/tasks", include_in_schema=True)
async def list_graded_tasks():
    """Expose ≥3 graded tasks for automated submission validation."""
    return _tasks_payload()


@app.get("/manifest/tasks", include_in_schema=True)
async def manifest_tasks_alias():
    """Alternate path some validators probe for task/grader manifests."""
    return _tasks_payload()


@app.get("/openenv.yaml", include_in_schema=False)
async def serve_openenv_yaml():
    """Serve the repo manifest verbatim (some pipelines fetch this URL)."""
    if not _openenv_yaml.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return FileResponse(_openenv_yaml, media_type="application/yaml")

if _static.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(_static)), 
        name="static"
    )
    
def main() -> None:
    import uvicorn
    # Port 7860 is required for HF Spaces
    port = int(os.environ.get("PORT", "7860"))
    # Disable reload in production containers (HF Spaces).
    reload_enabled = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=reload_enabled)
    
if __name__ == "__main__":
    main()
"""
FastAPI application for CodeCompleteEnv — Hugging Face Spaces deployment.

Exposes the OpenEnv interface over HTTP:

    POST /reset          → initial observation
    POST /step           → (observation, reward, done, info)
    GET  /state          → current observation
    GET  /tasks          → list of available tasks
    GET  /               → health check
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CodeCompleteEnv
from env.models import Action, Observation, Reward
from env.tasks import list_tasks

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeCompleteEnv",
    description=(
        "Reinforcement-learning environment for contextual code "
        "autocomplete.  Implements the OpenEnv step/reset/state API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory environment store keyed by task name
_envs: Dict[str, CodeCompleteEnv] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy_expression_complete"
    max_steps: int = 8


class StepRequest(BaseModel):
    task_name: str = "easy_expression_complete"
    completion: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health() -> Dict[str, str]:
    """Health-check / root endpoint."""
    return {"status": "ok", "env": "code_complete_env"}


@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    """List available tasks."""
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> Observation:
    """Reset (or create) the environment for a task and return the
    initial observation."""
    if req is None:
        req = ResetRequest()

    try:
        env = CodeCompleteEnv(
            task_name=req.task_name, max_steps=req.max_steps
        )
        obs = env.reset()
        _envs[req.task_name] = env
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state(task_name: str = "easy_expression_complete") -> Observation:
    """Return the current observation without advancing the clock."""
    env = _envs.get(task_name)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active environment for task '{task_name}'. Call /reset first.",
        )
    return env.state()


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    """Take one step in an active environment."""
    env = _envs.get(req.task_name)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No active environment for task '{req.task_name}'. "
                "Call /reset first."
            ),
        )

    action = Action(completion=req.completion)
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


# ---------------------------------------------------------------------------
# Uvicorn entry-point (for local dev / Docker CMD)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient, StepResult
from src.models import CodeAction, CodeObservation, CodeState


class CodeAssistClient(EnvClient[CodeAction, CodeObservation, CodeState]):
    """WebSocket client for the CRA code-assist environment."""

    def _step_payload(self, action: CodeAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeObservation]:
        obs_data = payload.get("observation", payload)
        obs = CodeObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeState:
        return CodeState(**payload)

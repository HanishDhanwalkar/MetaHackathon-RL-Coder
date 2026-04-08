from pydantic import Field
from typing import List
from openenv.core.env_server import Action, Observation, State


class CodeAction(Action):
    completion: str = Field(..., description="The code string to append")

class CodeObservation(Observation):
    code_context: str = Field(..., description="Current score Buffer")
    kg_context: List[str] = Field(default_factory=list, description="Optional Knowledge Graph hints")
    cursor_position: int = Field(..., description="Current cursor position")
    task_id: str = Field(..., description="active task or freeform")
    task_instruction: str = Field(..., description="Human readable objective")

class CodeState(State):
    current_task_id: str = Field(default="syntax-line", description="task id for this episode")
    # step_count: int = 0
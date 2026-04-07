from pydantic import Field
from openenv.core.env_server import Action, Observation, State
from typing import List

class CodeAction(Action):
    completion: str = Field(..., description="The code string to append")

class CodeObservation(Observation):
    code_context: str
    kg_context: List[str]
    cursor_position: int

class CodeState(State):
    current_task_id: str
    step_count: int = 0
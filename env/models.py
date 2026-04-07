"""
Typed Pydantic models for the CodeCompleteEnv environment.

Defines Observation, Action, and Reward schemas used by the
OpenEnv step()/reset()/state() interface.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    What the agent sees at each time-step.

    Attributes:
        cursor_file:      Name of the file the cursor is in.
        cursor_line:      1-based line number of the cursor.
        surrounding_code: Full source code in the active buffer (with a
                          ``__CURSOR__`` marker where completion is needed).
        kg_context:       Top-k relevant nodes from the knowledge graph.
        open_files:       List of filenames currently "open" in the editor.
        step_count:       How many steps have been taken so far.
    """

    cursor_file: str
    cursor_line: int
    surrounding_code: str
    kg_context: List[Dict[str, Any]] = Field(default_factory=list)
    open_files: List[str] = Field(default_factory=list)
    step_count: int = 0


class Action(BaseModel):
    """
    What the agent produces.

    Attributes:
        completion: The code snippet the agent wants inserted at the cursor.
    """

    completion: str


class Reward(BaseModel):
    """
    Multi-signal reward returned after each step.

    Every component is clamped to [0.0, 1.0].
    ``total`` is a weighted combination (see environment.py for weights).

    Attributes:
        ast_valid:      1.0 if the resulting code parses, else 0.0.
        type_correct:   Heuristic type-correctness score.
        style_match:    PEP-8 style adherence score.
        test_pass_rate: Fraction of unit-test cases that pass.
        user_signal:    Simulated user-acceptance heuristic.
        total:          Weighted aggregate of the component scores.
    """

    ast_valid: float = Field(default=0.0, ge=0.0, le=1.0)
    type_correct: float = Field(default=0.0, ge=0.0, le=1.0)
    style_match: float = Field(default=0.0, ge=0.0, le=1.0)
    test_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    user_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)

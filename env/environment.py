"""
CodeCompleteEnv — OpenEnv-compatible RL environment for code autocomplete.

Implements the full ``step() / reset() / state()`` interface with typed
Pydantic models and a dense multi-signal reward function.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Tuple

from .graders import (
    grade_ast_validity,
    grade_refactor,
    grade_style_match,
    grade_test_pass,
    grade_type_correct,
    grade_user_signal,
)
from .kg import CodeKnowledgeGraph
from .models import Action, Observation, Reward
from .tasks import get_task

# Reward component weights — must sum to 1.0
_WEIGHTS: Dict[str, float] = {
    "ast_valid": 0.30,
    "type_correct": 0.10,
    "style_match": 0.15,
    "test_pass_rate": 0.35,
    "user_signal": 0.10,
}


class CodeCompleteEnv:
    """Reinforcement-learning environment for contextual code completion.

    Lifecycle::

        env = CodeCompleteEnv(task_name="easy_expression_complete")
        obs = env.reset()

        for _ in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

        env.close()

    Parameters
    ----------
    task_name:
        One of ``easy_expression_complete``, ``medium_function_body``,
        or ``hard_refactor``.
    max_steps:
        Episode length cap.  Default 8.
    """

    def __init__(
        self,
        task_name: str = "easy_expression_complete",
        max_steps: int = 8,
    ) -> None:
        self.task_name = task_name
        self.max_steps = max_steps
        self._task_cfg = get_task(task_name)

        # Build knowledge graph once
        self._kg = CodeKnowledgeGraph()
        for node in self._task_cfg.get("kg_nodes", []):
            self._kg.add_node(
                name=node["name"],
                kind=node["kind"],
                context=node.get("context", ""),
            )

        # Internal state (populated by reset)
        self._current_code: str = ""
        self._original_code: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._last_reward: Reward | None = None
        self._rewards_history: list[float] = []
        self._observation: Observation | None = None

        # Auto-reset on construction so state() always works
        self.reset()

    # ------------------------------------------------------------------
    #  OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to the beginning of the task."""
        self._current_code = self._task_cfg["initial_code"]
        self._original_code = self._task_cfg["initial_code"]
        self._step_count = 0
        self._done = False
        self._last_reward = None
        self._rewards_history = []

        kg_ctx = self._kg.query(
            self._task_cfg.get("cursor_file", ""), top_k=5
        )
        self._observation = Observation(
            cursor_file=self._task_cfg["cursor_file"],
            cursor_line=self._task_cfg["cursor_line"],
            surrounding_code=self._current_code,
            kg_context=kg_ctx,
            open_files=self._task_cfg.get("open_files", []),
            step_count=0,
        )
        return self._observation

    def state(self) -> Observation:
        """Return the current observation (read-only peek)."""
        if self._observation is None:
            return self.reset()
        return self._observation

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Apply *action*, grade the result, and advance the clock.

        Returns
        -------
        observation : Observation
        reward      : Reward
        done        : bool
        info        : dict   (may include ``message`` or ``error``)
        """
        if self._done:
            assert self._last_reward is not None
            return (
                self._observation,  # type: ignore[return-value]
                self._last_reward,
                True,
                {"message": "Episode already finished."},
            )

        self._step_count += 1

        # 1. Apply the completion ----------------------------------------
        completed_code = self._apply_completion(action.completion)
        self._current_code = completed_code

        # 2. Grade -------------------------------------------------------
        reward = self._compute_reward(completed_code)
        self._last_reward = reward
        self._rewards_history.append(reward.total)

        # 3. Terminal check ----------------------------------------------
        done = False
        info: Dict[str, Any] = {}

        if reward.total >= 0.95:
            done = True
            info["message"] = "High score achieved"
        elif self._step_count >= self.max_steps:
            done = True
            info["message"] = "Max steps reached"

        self._done = done

        # 4. Build new observation ---------------------------------------
        kg_ctx = self._kg.query(
            action.completion[:60] if action.completion else "", top_k=5
        )

        self._observation = Observation(
            cursor_file=self._task_cfg["cursor_file"],
            cursor_line=self._task_cfg["cursor_line"],
            surrounding_code=self._current_code,
            kg_context=kg_ctx,
            open_files=self._task_cfg.get("open_files", []),
            step_count=self._step_count,
        )

        return self._observation, reward, done, info

    def close(self) -> None:
        """No-op cleanup; included for interface parity."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_completion(self, completion: str) -> str:
        """Insert the agent's completion into the source code."""
        marker = self._task_cfg.get("cursor_marker")
        if marker and marker in self._original_code:
            return self._original_code.replace(marker, completion)
        # Hard / full-file tasks: the completion *is* the new file
        return completion

    def _compute_reward(self, code: str) -> Reward:
        """Run all graders and return a weighted ``Reward``."""
        ast_score = grade_ast_validity(code)
        type_score = grade_type_correct(code)
        style_score = grade_style_match(code)
        test_score = grade_test_pass(code, self._task_cfg.get("test_cases", []))
        user_score = grade_user_signal(code, self._original_code)

        # Hard-task: blend refactor score into test_pass_rate
        refactor_cfg = self._task_cfg.get("refactor_target")
        if refactor_cfg:
            refactor_score = grade_refactor(
                code,
                old_name=refactor_cfg["old_name"],
                new_name=refactor_cfg["new_name"],
                expected_replacements=refactor_cfg["expected_replacements"],
            )
            test_score = test_score * 0.5 + refactor_score * 0.5

        total = (
            _WEIGHTS["ast_valid"] * ast_score
            + _WEIGHTS["type_correct"] * type_score
            + _WEIGHTS["style_match"] * style_score
            + _WEIGHTS["test_pass_rate"] * test_score
            + _WEIGHTS["user_signal"] * user_score
        )

        # Destructive-completion penalty
        if len(code.strip()) < len(self._original_code.strip()) * 0.20:
            total *= 0.3

        total = max(0.0, min(1.0, round(total, 4)))

        return Reward(
            ast_valid=round(ast_score, 4),
            type_correct=round(type_score, 4),
            style_match=round(style_score, 4),
            test_pass_rate=round(test_score, 4),
            user_signal=round(user_score, 4),
            total=total,
        )

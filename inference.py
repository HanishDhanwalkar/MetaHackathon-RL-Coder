"""
Inference script for CodeCompleteEnv.

Runs a baseline LLM agent against all three tasks (easy, medium, hard)
using the OpenAI-compatible chat completions API.

Environment variables
---------------------
HF_TOKEN / API_KEY / OPENAI_API_KEY : str
    API key for the LLM provider.
API_BASE_URL : str
    Chat-completions endpoint (default: HuggingFace router).
MODEL_NAME : str
    Model identifier (default: Qwen/Qwen2.5-72B-Instruct).

STDOUT format
-------------
[START] task=<task_name> env=code_complete_env model=<model_name>
[STEP]  step=<n> action=<string> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from typing import List

from openai import OpenAI

from env.environment import CodeCompleteEnv
from env.models import Action
from env.tasks import list_tasks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "code_complete_env"
MAX_STEPS: int = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip markdown fences if the model wrapped its answer in them."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Sometimes models return just the code
    return text.strip()


def _oneline(text: str, max_len: int = 120) -> str:
    """Collapse *text* to a single safe line for stdout logging."""
    s = text.replace("\n", "\\n").replace("\r", "")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _build_system_prompt() -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given incomplete Python source code with a cursor "
        "position and contextual information. Your job is to produce "
        "the best possible code completion.\n\n"
        "Rules:\n"
        "1. Output ONLY the replacement code — no explanations, no markdown.\n"
        "2. Match the surrounding style (indentation, naming, etc.).\n"
        "3. Make sure the result is syntactically valid Python.\n"
    )


def _build_user_prompt(obs, task_name: str, task_cfg: dict) -> str:
    """Build the user-turn prompt from the current observation."""
    parts: list[str] = []

    # Task instruction
    if task_cfg.get("cursor_marker"):
        parts.append(
            f"Complete the code at the cursor position marked with "
            f"`{task_cfg['cursor_marker']}` in file `{obs.cursor_file}`.\n"
        )
    else:
        # Hard / refactor task
        refactor = task_cfg.get("refactor_target", {})
        parts.append(
            f"Refactor the code below by renaming the function "
            f"`{refactor.get('old_name', '')}` to "
            f"`{refactor.get('new_name', '')}` everywhere it appears.\n"
            f"Output the COMPLETE refactored file — every line.\n"
        )

    # Source code
    parts.append("--- source code ---")
    parts.append(obs.surrounding_code)
    parts.append("--- end ---\n")

    # KG context
    if obs.kg_context:
        parts.append("Relevant code elements:")
        for item in obs.kg_context:
            parts.append(
                f"  - {item['name']} ({item['kind']}): {item['context']}"
            )
        parts.append("")

    # Feedback from previous step
    if obs.step_count > 0:
        parts.append(
            f"(This is attempt {obs.step_count + 1}. "
            f"Previous code was not fully correct — try again.)\n"
        )

    if task_cfg.get("cursor_marker"):
        parts.append(
            "Respond with ONLY the code that replaces "
            f"`{task_cfg['cursor_marker']}`. No markdown fences."
        )
    else:
        parts.append(
            "Respond with the COMPLETE refactored source file. "
            "No markdown fences, no explanations."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, client: OpenAI) -> float:
    """Run a single task and return the best score achieved."""
    from env.tasks import get_task

    task_cfg = get_task(task_name)
    env = CodeCompleteEnv(task_name=task_name, max_steps=MAX_STEPS)

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    obs = env.reset()
    rewards: List[float] = []
    done = False
    step = 0
    last_error: str | None = None

    while not done and step < MAX_STEPS:
        step += 1

        # --- LLM call ---------------------------------------------------
        prompt = _build_user_prompt(obs, task_name, task_cfg)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            completion = _extract_code(raw)
        except Exception as exc:
            completion = ""
            last_error = str(exc)

        # --- Environment step -------------------------------------------
        action = Action(completion=completion)
        try:
            obs, reward, done, info = env.step(action)
            reward_val = reward.total
            last_error = info.get("error")
        except Exception as exc:
            reward_val = 0.0
            done = True
            last_error = str(exc)

        rewards.append(reward_val)

        err_str = f"error={last_error}" if last_error else "error=null"
        print(
            f"[STEP] step={step} "
            f"action={_oneline(completion)} "
            f"reward={reward_val:.2f} "
            f"done={'true' if done else 'false'} "
            f"{err_str}"
        )

    env.close()

    final_score = max(rewards) if rewards else 0.0
    success = final_score >= 0.7
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} score={final_score:.2f} rewards={rewards_str}"
    )
    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print(
            "ERROR: No API key found. Set HF_TOKEN, API_KEY, or "
            "OPENAI_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    tasks = list_tasks()
    scores: dict[str, float] = {}

    for task_name in tasks:
        try:
            scores[task_name] = run_task(task_name, client)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            scores[task_name] = 0.0
            print(
                f"[END] success=false steps=0 score=0.00 rewards=0.00"
            )

    print("\n=== Summary ===")
    for name, score in scores.items():
        print(f"  {name}: {score:.2f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  average:  {avg:.2f}")


if __name__ == "__main__":
    main()

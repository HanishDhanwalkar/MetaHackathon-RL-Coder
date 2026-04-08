import os
import re
import sys

from typing import List, Optional, Any

import ollama
from dotenv import load_dotenv
import logging 

from src.code_assist_env import CodeAssistEnv
from src.models import CodeAction, CodeObservation
from src.rl_agent import RLCompletionAgent
from src.workspace_kg import WorkspaceKG


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
HF_TOKEN = os.getenv("HF_TOKEN")


client = ollama.Client(
    host=API_BASE_URL.rstrip("/"),
)
env = CodeAssistEnv()
kg = WorkspaceKG()
policy = RLCompletionAgent()


def clean_suggests(raw: str) -> dict[str, Any]:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t

def sync_workspace(source: str) -> dict[str, Any]:
    meta = kg.update(source)
    return {
        "major_changed": meta.get("major_changes", False),
        "symbol_count": meta.get("symbol_count", 0),
        "import_count": meta.get("import_count", 0),
        "valid_parse": meta.get("valid_parse", False),
    }
    
# TODO: Make this openaAI complatible
def call_llm(
    prompt: str, 
    sys_prompt: str,
    model_name: str,
    options: dict[str, Any],
) -> dict[str, Any]:

    # OLLAMAto Openai

    res = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        options=options
    )
    
    return res

def get_completion(
    context: str,
    c_offset: Optional[int] = None,
) -> dict[str, Any]:
    kg_meta = kg.update(context)
    
    cur = (
        c_offset
        if c_offset is not None
        else len(context) if context else 0
    )
    cur = max(0, min(cur, len(context)))
    kg_lines = kg.context_lines(context, cur)
    
    
    obs: CodeObservation = env.reset(
        code_context=context,
        kg_context=kg_lines,
        cursor_offset=cur
    )
    
    before = context[:cur]
    after = context[cur:]
    
    kg_block = "\n".join(kg_lines) if kg_lines else "(no graph edges yet)"
    prompt =(
        f"{obs.task_instructions}\n\n"
        f"Workspace Knowledge (use for naming, imports, and local calls): \n{kg_block}\n\n"
        "Insert ONLY the raw code snippet at cursor location:\n\n"
        f"<before_cursor>\n{before}\n</before_cursor>\n"
        f"<after_cursor>\n{after}\n</after_cursor>\n"
    )
    
    response = call_llm(
        model_name=MODEL_NAME,
        sys_prompt=policy.sys_msg(),
        prompt=prompt,
        options=policy.ollama_options(), # TODO: Fix for OpenAI
    )
    
    suggestion = clean_suggests(response["message"]["content"])
    out = env.step(CodeAction(completion=suggestion))
    policy.obserce_reward(
        float(out["reward"]) if out["reward"] is not None else 0.0
    )
    
    err = None
    if out.metadata:
         err = out.metadata.get("last_action_err")
         
    return{
        "completion": suggestion,
        "reward": out["reward"],
        "done": out["done"],
        "error": err,
        "rl_ema_reward": round(policy.ema_reward(), 3),
        "rl_trend": round(policy.trend(), 3),
        "kg_major_changed": kg_meta.get("major_changes"),
        "kg_symbols": kg_meta.get("symbol_count"),
        "cursor_after_insert": out["cursor_position"]
    }
    
def err_token(msg: str|None)-> str:
    if msg is None or msg == "":
        return "null"
    else:
        return msg.replace("\n", " ").replace("\r", " ")
    
def run_graded_baseline() -> None:
    rewards: List[float] = []
    benchmark = "code_assist_env"
    print(
        f"[START] task=graded-tasks env={benchmark} model={MODEL_NAME}",
        flush=True,
    )
    step_n = 0
    envs: List[CodeAssistEnv] = []
    try:
        for task_id in ("syntax-line", "import-fix", "docstring-stub"):
            env = CodeAssistEnv()
            envs.append(env)
            obs = env.reset(task_id=task_id)
            prompt = (
                f"{obs.task_instruction}\n\n"
                "Return only text to append to the end of this Python file "
                "(no markdown fences):\n"
                f"{obs.code_context}"
            )
            err_raw: str | None = None
            suggestion = ""
            try:
                response = call_llm(
                    model_name=MODEL_NAME,
                    sys_prompt=policy.sys_msg(),
                    prompt=prompt,
                    options=policy.ollama_options(),
                )
                suggestion = clean_suggests(response["message"]["content"])
            except Exception as exc:
                err_raw = str(exc)

            step_obs = env.step(CodeAction(completion=suggestion))
            policy.observe_reward(
                float(step_obs.reward) if step_obs.reward is not None else 0.0
            )
            step_n += 1
            r = float(step_obs.reward) if step_obs.reward is not None else 0.0
            rewards.append(r)
            done_s = str(bool(step_obs.done)).lower()
            meta_err = None
            if step_obs.metadata:
                meta_err = step_obs.metadata.get("last_action_error")
            combined_err = err_raw or meta_err
            action_lit = repr(suggestion)
            print(
                f"[STEP] step={step_n} action={action_lit} reward={r:.2f} "
                f"done={done_s} error={err_token(combined_err)}",
                flush=True,
            )
    finally:
        for e in envs:
            e.close()
        ok = bool(rewards) and all(x >= 0.99 for x in rewards)
        rfmt = ",".join(f"{float(x):.2f}" for x in rewards)
        print(
            f"[END] success={str(ok).lower()} steps={step_n} rewards={rfmt}",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        run_graded_baseline()
    else:
        print(get_completion("import os\ndef list_files():\n    "))
 
import os
import re
import sys

from typing import List, Optional, Any

# import ollama  # Commented out: replaced with OpenAI
from openai import OpenAI
from dotenv import load_dotenv
import logging 

from src.code_assist_env import CodeAssistEnv
from src.models import CodeAction, CodeObservation
from src.rl_agent import RLCompletionAgent
from src.workspace_kg import WorkspaceKG


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()


client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)
env = CodeAssistEnv()
kg = WorkspaceKG()
policy = RLCompletionAgent()


def clean_suggests(raw: str) -> str:
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
    
def call_llm(
    prompt: str, 
    sys_prompt: str,
    model_name: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Call OpenAI-compatible chat completions API (routed via Hugging Face)."""
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=options.get("temperature", 0.2),
            top_p=options.get("top_p", 0.9),
            max_tokens=options.get("max_tokens", 100),
        )
        
        # Return a dict matching the shape the rest of the code expects
        return {
            "message": {"content": res.choices[0].message.content}
        }
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {
            "message": {"content": "pass"}
        }

def get_completion(
    context: str,
    c_offset: Optional[int] = None,
) -> dict[str, Any]:
    try:
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
        prompt = (
            "You are an expert Python developer.\n"
            "Return only the missing code.\n\n"
            f"Context:\n{kg_block}\n\n"
            f"Code:\n{before}\n\n"
            "Complete:"
        )
        
        response = call_llm(
            model_name=MODEL_NAME,
            sys_prompt=policy.sys_msg(),
            prompt=prompt,
            options=policy.openai_options(),
        )
        
        suggestion = clean_suggests(response["message"]["content"])
        out = env.step(CodeAction(completion=suggestion))
        policy.observe_reward(
            float(out.reward) if out.reward is not None else 0.0
        )
        
        err = None
        if out.metadata:
             err = out.metadata.get("last_action_err")
             
        return {
            "completion": suggestion,
            "reward": out.reward,
            "done": out.done,
            "error": err,
            "rl_ema_reward": round(policy.ema_reward(), 3),
            "rl_trend": round(policy.trend(), 3),
            "kg_major_changed": kg_meta.get("major_changes"),
            "kg_symbols": kg_meta.get("symbol_count"),
            "cursor_after_insert": out.cursor_position
        }
    except Exception as exc:
        logger.error("get_completion failed: %s", exc)
        return {
            "completion": "pass",
            "reward": 0.0,
            "done": False,
            "error": str(exc),
            "rl_ema_reward": round(policy.ema_reward(), 3),
            "rl_trend": round(policy.trend(), 3),
            "kg_major_changed": False,
            "kg_symbols": 0,
            "cursor_after_insert": 0
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
            task_env = CodeAssistEnv()
            envs.append(task_env)
            obs = task_env.reset(task_id=task_id)
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
                    options=policy.openai_options(),
                )
                suggestion = clean_suggests(response["message"]["content"])
            except Exception as exc:
                err_raw = str(exc)

            step_obs = task_env.step(CodeAction(completion=suggestion))
            policy.observe_reward(
                float(step_obs.reward) if step_obs.reward is not None else 0.0
            )
            step_n += 1
            r = float(step_obs.reward) if step_obs.reward is not None else 0.0
            rewards.append(r)
            done_s = str(bool(step_obs.done)).lower()
            meta_err = None
            if step_obs.metadata:
                meta_err = (
                    step_obs.metadata.get("last_action_err")
                    or step_obs.metadata.get("last_action_error")
                )
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
        score = 0.0
        if rewards:
            score = sum(rewards) / len(rewards)
            score = min(max(score, 0.0), 1.0)
        ok = bool(rewards) and score >= 0.99
        rfmt = ",".join(f"{float(x):.2f}" for x in rewards)
        print(
            f"[END] success={str(ok).lower()} steps={step_n} score={score:.2f} rewards={rfmt}",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        run_graded_baseline()
    else:
        print(get_completion("import os\ndef list_files():\n    "))
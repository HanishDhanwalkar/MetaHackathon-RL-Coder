import os
import re
import sys

from typing import List, Optional, Any, Tuple

from openai import OpenAI
from openai import APIStatusError
from dotenv import load_dotenv
import logging

from src.code_assist_env import CodeAssistEnv
from src.models import CodeAction, CodeObservation
from src.rl_agent import RLCompletionAgent
from src.workspace_kg import WorkspaceKG


load_dotenv()

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-Next").strip()
HF_TOKEN = (
    os.getenv("HF_TOKEN", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
)

# Set to 1 to skip remote LLM and use local heuristics only (fast, no credits).
FORCE_OFFLINE = os.getenv("OFFLINE_COMPLETIONS", "").strip().lower() in {"1", "true", "yes"}

if HF_TOKEN == "":
    logger.warning(
        "No API token found. Set HF_TOKEN (preferred) or OPENAI_API_KEY."
    )


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


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _billing_or_quota(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in (401, 402, 403, 429):
        return True
    msg = str(exc).lower()
    return "credit" in msg or "payment" in msg or "quota" in msg or "depleted" in msg


class _RemoteErr(Exception):
    """Carries HTTP status for billing detection when reconstructing from a dict."""

    def __init__(self, message: str, status_code: Optional[int]) -> None:
        super().__init__(message)
        self.status_code = status_code


def _billing_error_from_llm_response(resp: dict[str, Any]) -> bool:
    if resp.get("ok") is not False:
        return False
    msg = str(resp.get("error") or "")
    code = resp.get("status_code")
    sc = code if isinstance(code, int) else None
    return _billing_or_quota(_RemoteErr(msg, sc))


def _cursor_lines(before: str, after: str) -> Tuple[str, str]:
    line_before = before.split("\n")[-1] if before else ""
    line_after = (after.split("\n")[0] if after else "") or ""
    return line_before, line_after


def _heuristic_completion(before: str, after: str, full: str) -> Tuple[str, int]:
    """Fast local completion when the LLM is unavailable. Returns (insert_text, delete_after)."""
    line_before, _line_after = _cursor_lines(before, after)
    lb = line_before.rstrip()

    # Balance delimiters (cheap, file-wide — good enough for ghost text).
    if lb.endswith("(") and full.count("(") > full.count(")"):
        return ")", 0
    if lb.endswith("[") and full.count("[") > full.count("]"):
        return "]", 0
    if lb.endswith("{") and full.count("{") > full.count("}"):
        return "}", 0

    if re.search(r"\breturn\s+$", line_before):
        return "None", 0
    if re.search(r"\breturn$", lb):
        return " None", 0

    if lb.endswith(":"):
        m = re.match(r"^(\s*)", line_before)
        ws = m.group(1) if m else ""
        return f"\n{ws}    pass", 0

    return "", 0


def _merge_insert(line_after: str, suggestion: str) -> Tuple[str, int]:
    """Trim insert text and optionally delete the identifier suffix after the cursor."""
    if not suggestion:
        return "", 0
    if "\n" in suggestion:
        return suggestion, 0
    m = re.match(r"^[A-Za-z0-9_]+", line_after)
    token_right = m.group(0) if m else ""
    if not token_right:
        return suggestion, 0
    if suggestion.startswith(token_right):
        return suggestion[len(token_right) :], len(token_right)
    if token_right and not suggestion.startswith(token_right):
        cap = min(len(token_right), 80)
        return suggestion, cap
    return suggestion, 0


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
    if FORCE_OFFLINE:
        return {"message": {"content": ""}, "ok": False, "error": "offline_mode", "status_code": None}

    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=options.get("temperature", 0.2),
            top_p=options.get("top_p", 0.9),
            max_tokens=options.get("max_tokens", 64),
        )
        content = res.choices[0].message.content
        return {
            "message": {"content": content},
            "ok": True,
            "error": None,
            "status_code": 200,
        }
    except APIStatusError as exc:
        code = getattr(exc, "status_code", None)
        if _billing_or_quota(exc):
            logger.error(
                "LLM unavailable (%s): %s. Set OPENAI_API_KEY for OpenAI, "
                "add HF Inference credits, or OFFLINE_COMPLETIONS=1 for local heuristics.",
                code,
                exc,
            )
        else:
            logger.error("LLM call failed: %s", exc)
        return {
            "message": {"content": ""},
            "ok": False,
            "error": str(exc),
            "status_code": code,
        }
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {
            "message": {"content": ""},
            "ok": False,
            "error": str(exc),
            "status_code": _status_code(exc),
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
            cursor_offset=cur,
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

        raw = response["message"].get("content") or ""
        suggestion = clean_suggests(raw)
        billing_error = _billing_error_from_llm_response(response)

        source = "llm"
        delete_after = 0

        if (not suggestion) or (response.get("ok") is False):
            h_text, h_del = _heuristic_completion(before, after, context)
            if h_text or h_del:
                suggestion = h_text
                delete_after = h_del
                source = "heuristic"
            elif response.get("ok") is False:
                source = "none"

        if suggestion:
            merged, d_after = _merge_insert(
                (after.split("\n")[0] if after else "") or "", suggestion
            )
            suggestion = merged
            delete_after = max(delete_after, d_after)

        out = env.step(CodeAction(completion=suggestion or ""))
        policy.observe_reward(float(out.reward) if out.reward is not None else 0.0)

        err = None
        if out.metadata:
            err = out.metadata.get("last_action_err")

        api_err = None if response.get("ok") else response.get("error")
        if FORCE_OFFLINE:
            api_err = (
                "OFFLINE_COMPLETIONS=1: remote LLM disabled; using local heuristics."
            )
        elif billing_error:
            api_err = (
                "Inference credits exhausted (402). Add HF prepaid credits, use PRO, "
                "set OPENAI_API_KEY with API_BASE_URL=https://api.openai.com/v1, "
                "or OFFLINE_COMPLETIONS=1 for local-only suggestions."
            )

        return {
            "completion": suggestion,
            "delete_after": delete_after,
            "reward": out.reward,
            "done": out.done,
            "error": err,
            "api_error": api_err,
            "billing_error": bool(billing_error),
            "source": source,
            "rl_ema_reward": round(policy.ema_reward(), 3),
            "rl_trend": round(policy.trend(), 3),
            "kg_major_changed": kg_meta.get("major_changes"),
            "kg_symbols": kg_meta.get("symbol_count"),
            "cursor_after_insert": out.cursor_position,
        }
    except Exception as exc:
        logger.error("get_completion failed: %s", exc)
        return {
            "completion": "",
            "delete_after": 0,
            "reward": 0.0,
            "done": False,
            "error": str(exc),
            "api_error": str(exc),
            "billing_error": False,
            "source": "none",
            "rl_ema_reward": round(policy.ema_reward(), 3),
            "rl_trend": round(policy.trend(), 3),
            "kg_major_changed": False,
            "kg_symbols": 0,
            "cursor_after_insert": 0,
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
            response = call_llm(
                model_name=MODEL_NAME,
                sys_prompt=policy.sys_msg(),
                prompt=prompt,
                options=policy.openai_options(),
            )
            suggestion = clean_suggests(response["message"].get("content") or "")
            if not response.get("ok"):
                err_raw = str(response.get("error"))

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
    # Evaluator runs `python inference.py` with no args — must emit
    # [START] / [STEP] / [END] on stdout only.
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        print(get_completion("import os\ndef list_files():\n    "))
    else:
        run_graded_baseline()
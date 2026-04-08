

import ast
import re
import uuid
from typing import Any, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .models import CodeAction, CodeObservation, CodeState

MAX_STEPS = 100
# MAX_CODE_LENGTH = 1000
MAX_COMPLETION_LENGTH = 2000
REPEAT_WINDOW = 3

TASK_LIBRARY = {
    "syntax-line": {
        "difficulty": "easy",
        "instruction": "Finish the line so the module is valid Python (balanced parentheses).",
        "starter": "result = (40 + 2"
    },
    "import-fix": {
        "difficulty": "medium",
        "instruction": "Add the missing import so the file runs.",
        "starter": (
            "def load_config(path):\n"
            "   with open(path, 'r') as f:\n"
            "       return json.load(f)\n"
        ),
    },
    "docstring-stub": {
        "difficulty": "hard",
        "instruction": "Give the function a docstring and a body with a return statement.",
        "starter": "def moving_average(values, window):\n"
    },
}


class CodeAssistEnv(Environment[CodeAction, CodeObservation, CodeState]):

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, task: str = "syntax-line"):
        super().__init__()
        self._code = ""
        self._cursor = ""
        self._kg_hints: List[str] = []
        self._active_task = "syntax-line"
        self._step_idx = 0
        self._eps_id: Optional[str] = None
        self._last_action_err = ""
        self._recent_completions: List[str] = []

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="code-assist_env",
            description="Code Completion Environment",  # TODO: ADD MORE
            version="0.1.0",
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeObservation:
        self._reset_rubric()
        self._step_idx = 0
        self._eps_id = episode_id or str(uuid.uuid4())
        self._last_action_err = ""
        self._recent_completions.clear()

        task_id = kwargs.get("task_id", "syntax-line")
        code_context = kwargs.get("code_context")
        kg = kwargs.get("kg_context")
        self._kg_hints = list(kg) if kg else []

        if code_context:
            self._active_task = "freeform"
            self._code = str(code_context)

            co = kwargs.get("cursor_offset")
            if co is None:
                self._cursor = len(self._code)
            else:
                self._cursor = max(0, min(int(co), len(self._code)))

        elif task_id in TASK_LIBRARY:
            self._active_task = task_id
            self._code = TASK_LIBRARY[task_id]["starter"]

        else:
            self._active_task = "syntax-line"
            self._code = TASK_LIBRARY["syntax-line"]["starter"]

        if self._active_task != "freeform":
            self._cursor = len(self._code)

        return self._observations(reward=0.0, done=False)
    
    def step(self, action: CodeAction, timeout_s: Optional[float] = None, **kwargs: Any) -> CodeObservation:
        self._last_action_err = ""
        raw = action.completion if action.completion else ""
        
        if  raw > MAX_COMPLETION_LENGTH:
            self._last_action_err = "Completion too long"
            self._step_idx += 1
            return self._observations(reward=0.25, done=self._forced_done())
        
        if not raw.strip():
            self._last_action_err = "Empty completion"
            self._step_idx += 1
            return self._observations(reward=-0.05, done=self._forced_done())
        
        self._recent_completions.append(raw)
        if len(self._recent_completions) > REPEAT_WINDOW:
            tail = self._recent_completions[-REPEAT_WINDOW:]
            if tail[0] == tail[1] == tail[2]:
                self._last_action_err = "Repeated completion"
        
        if self._active_task == "freeform":
            before = self._code[:self._cursor]
            after = self._code[self._cursor:]
            self._code = before + raw + after
            self._cursor += len(raw)
        
        else:
            self._code += raw
            self._cursor += len(raw)
            
        base = self._grade()
        repeat_penal = 0.2 if self._last_action_err == "Repeated completion" else 0.0
        reward = max(0.0, min(1.0, base - repeat_penal))
        
        self._step_idx += 1
        objective_met = reward > 0.90
        out_of_scope = self._step_idx >= MAX_STEPS
        done = objective_met or out_of_scope
        
        return self._observations(reward=reward, done=done)
    
    @property
    def step(self) -> CodeState:
        return CodeState(
            episode_id=self._eps_id,
            current_task_id=self._active_task,
            step_count=self._step_idx
        )
        # if raw in self._recent_completions:
        #     self._last_action_err = "Repeated completion"
        #     self._step_idx += 1
        #     return self._observations(reward=-0.05, done=self._forced_done())
        
        # try:
        #     new_code = self._code[:self._cursor] + raw + self._code[self._cursor:]
        #     ast.parse(new_code)
        #     self._code = new_code

    def _instruction(self) -> str:
        if self._active_task == "freeform":
            return "Keep the editor content valid, idiomatic Python code."
        return TASK_LIBRARY.get(self._active_task, TASK_LIBRARY["syntax-line"])["instruction"]
    
    def _forced_done(self) -> bool:
        return self._step_idx >= MAX_STEPS

    def _observations(self, reward: float, done: bool) -> CodeObservation:
        return CodeObservation(
            code_context=self._code,
            kg_context=self._kg_hints,
            cursor_position=self._cursor,
            task_id=self._active_task,
            task_instruction=self._instruction(),
            reward=round(reward, 2),
            done=done,
            metadata={
                "last_action_err": self._last_action_err or None,
                "difficulty": (
                    TASK_LIBRARY.get(self._active_task, {}).get(
                        "difficulty", "n/a")
                    if self._active_task != "freeform"
                    else "n/a"
                ),
            }
        )

    def _grade_freeform(self):
        try:
            compile(self._code, "<editor>", "exec")
        except SyntaxError:
            return 0.15
        else:
            return 1.0
    
    def _grade_syntax_line(self):
        score = 0.0
        if self._code.count("(") != self._code.count(")"):
            score += 0.45
            
        try:
            compile(self._code, "<task>", "exec")
            score += 0.45
        except SyntaxError:
            pass
        return min(1.0, score)
    
    def _grade_import_fix(self):
        score = 0.0
        if re.search(r"\s import\s + json\b", self._code, re.MULTILINE):
            score += 0.45
            
        try:
            compile(self._code, "<task>", "exec")
            score += 0.45
        except SyntaxError:
            pass
        return min(1.0, score)
    
    def _grade_docstring(self):
        score = 0.0
        try:
            tree = ast.parse(self._code)
        except SyntaxError:
            return 0.1
        
        fn: Optional[ast.FunctionDef] = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "moving_average":
                fn = node
                break
        if fn is None:
            return 0.15
        
        doc =fn.get_docstring(fn)
        
        if doc and len(doc.strip()) > 12:
            score += 0.45
            
        body = [n for n in fn.body if isinstance(n, ast.Expr)]
        
        if body and any(isinstance(b, ast.Return) for b in body):
            score += 0.45
        elif body:
            score += 0.15
        
        try: 
            compile(self._code, "<task>", "exec")
            score = min(1.0, score+0.05)
            
        except SyntaxError:
            score*=0.5
        return min(1.0, score)
# CodeCompleteEnv

> A production-ready **OpenEnv** reinforcement-learning environment where an AI agent learns **contextual Python code autocomplete** through dense rewards from syntax correctness, test passing, style matching, and simulated user feedback.

---

## 🎯 Problem Description

Code autocomplete is one of the highest-impact tasks in developer tooling.  
Modern AI assistants must produce completions that are:

1. **Syntactically valid** — the code must parse.
2. **Semantically correct** — it must do the right thing.
3. **Stylistically consistent** — it must match the surrounding code.
4. **Acceptable to the user** — concise, non-destructive, helpful.

**CodeCompleteEnv** models this as a multi-step RL episode: the agent observes incomplete source code with cursor context and knowledge-graph hints, produces a completion, and receives a **multi-signal reward** balancing all four criteria above.

---

## 📐 Observation Space

| Field              | Type                  | Description                                         |
| ------------------ | --------------------- | --------------------------------------------------- |
| `cursor_file`      | `str`                 | Name of the file the cursor is in                   |
| `cursor_line`      | `int`                 | 1-based line number of the cursor                   |
| `surrounding_code` | `str`                 | Full source with `__CURSOR__` marker                |
| `kg_context`       | `List[Dict[str,Any]]` | Top-k knowledge-graph nodes (name, kind, context)   |
| `open_files`       | `List[str]`           | Other files "open" in the editor                    |
| `step_count`       | `int`                 | Steps taken so far                                  |

---

## 🎮 Action Space

| Field        | Type  | Description                               |
| ------------ | ----- | ----------------------------------------- |
| `completion` | `str` | Code snippet the agent inserts at cursor  |

---

## 🏆 Reward Design

The reward is **dense** (not binary) — a weighted sum of five signals, each in \[0.0, 1.0\]:

| Component        | Weight | Description                                      |
| ---------------- | ------ | ------------------------------------------------ |
| `ast_valid`      | 0.30   | Does the completed code parse as valid Python?   |
| `test_pass_rate` | 0.35   | Fraction of unit-test cases that pass            |
| `style_match`    | 0.15   | PEP-8 adherence (line length, indent, naming)    |
| `type_correct`   | 0.10   | Heuristic type-correctness (return statements)   |
| `user_signal`    | 0.10   | Simulated user acceptance (quality proxies)      |

**Additional shaping:**
- **Partial credit** on almost-parseable code (truncated completions).
- **Penalty** for destructive completions (stripping >80 % of original code).
- **Partial credit** for partially-correct refactors (some-but-not-all renames).

---

## 📝 Tasks

### Easy — `easy_expression_complete`

Complete a missing `return` expression in a simple function (`calculate_area`).

**Grading:** AST parse success + test-case evaluation (5 tests).

**Expected difficulty:** Trivially solved by any competent LLM in 1 step.

### Medium — `medium_function_body`

Generate the full body of `flatten_list` from its signature and docstring.

**Grading:** 6 unit tests covering edge cases (empty list, deep nesting, mixed types).

**Expected difficulty:** Requires understanding recursion; most models need 1–2 steps.

### Hard — `hard_refactor`

Rename the function `calc` to `calculate_total_price` across an entire file with 6 call sites, then verify that all dependent functions still work.

**Grading:** Refactor-completeness score (all old references removed, all new references present) **×** unit-test pass rate (5 tests across 3 functions).

**Expected difficulty:** Requires careful whole-file edit; mistakes compound.

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (optional, for containerised runs)

### Local Setup

```bash
# Clone / navigate to the project root
cd code_complete_env

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the API Server Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Then visit `http://localhost:7860/docs` for the interactive Swagger UI.

### Run with Docker

```bash
docker build -t code-complete-env .
docker run -p 7860:7860 code-complete-env
```

### Run the Inference Script

```bash
export HF_TOKEN="your-api-key"                                   # or OPENAI_API_KEY
export API_BASE_URL="https://router.huggingface.co/v1"            # optional
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"                    # optional

python inference.py
```

---

## 🤗 Hugging Face Spaces Deployment

1. Create a new **Docker** Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Push this repository to the Space repo.
3. Set Space secrets: `HF_TOKEN`.
4. The Dockerfile auto-builds and starts the API on port 7860.
5. Tag the Space with `openenv`.

**Verify:**

```bash
curl -X POST https://<your-space>.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "easy_expression_complete"}'
```

Should return HTTP 200 with the initial `Observation` JSON.

---

## 📊 Example Baseline Scores

Results from `Qwen/Qwen2.5-72B-Instruct` via HuggingFace inference:

| Task                       | Steps | Score  | Result    |
| -------------------------- | ----- | ------ | --------- |
| `easy_expression_complete` | 1     | ~0.90  | ✅ success |
| `medium_function_body`     | 1–2   | ~0.82  | ✅ success |
| `hard_refactor`            | 1–3   | ~0.75  | ✅ success |
| **Average**                |       | **~0.82** |        |

*Scores may vary slightly depending on API temperature and model version.*

---

## 📂 Project Structure

```
code_complete_env/
├── env/
│   ├── __init__.py        # Package exports
│   ├── environment.py     # CodeCompleteEnv (step/reset/state)
│   ├── models.py          # Pydantic: Observation, Action, Reward
│   ├── graders.py         # Deterministic grading functions
│   ├── tasks.py           # Task definitions (easy/medium/hard)
│   └── kg.py              # Lightweight NetworkX knowledge graph
├── inference.py           # Baseline LLM agent script
├── openenv.yaml           # OpenEnv spec metadata
├── app.py                 # FastAPI server (HF Spaces)
├── Dockerfile             # Container build
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Environment Variables

| Variable        | Required | Default                                    | Description               |
| --------------- | -------- | ------------------------------------------ | ------------------------- |
| `HF_TOKEN`      | Yes*     | —                                          | API key for LLM calls     |
| `API_BASE_URL`  | No       | `https://router.huggingface.co/v1`         | LLM endpoint              |
| `MODEL_NAME`    | No       | `Qwen/Qwen2.5-72B-Instruct`               | Model identifier          |

\* Or `API_KEY` / `OPENAI_API_KEY`.

---

## License

MIT

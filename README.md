---
title: Hanishian Test Space
emoji: ⚡
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
tags:
  - openenv
---

# CRA – Contextual Reinforcement Autocomplete

An RL-powered Python code-completion environment built on the **OpenEnv** specification.

## Environment Overview

CRA creates a code-editing sandbox where an RL agent receives partial Python
source and must emit a completion that makes the code valid, idiomatic, and
functionally correct. A **Knowledge Graph** (AST-level) and an **EMA-based
policy adapter** provide dense, curriculum-aware reward signals.

## Action & Observation Spaces

| Field | Type | Description |
|---|---|---|
| **Action** — `CodeAction.completion` | `str` | Raw code string to insert at cursor |
| **Observation** — `code_context` | `str` | Current editor buffer |
| **Observation** — `kg_context` | `List[str]` | KG hints (imports, symbols, calls) |
| **Observation** — `cursor_position` | `int` | Byte offset of cursor |
| **Observation** — `task_instruction` | `str` | Human-readable objective |
| **Observation** — `reward` | `float` | Dense reward ∈ [0, 1] |
| **Observation** — `done` | `bool` | Episode termination flag |

## Tasks (Easy → Hard)

| ID | Difficulty | Objective | Grading |
|---|---|---|---|
| `syntax-line` | Easy | Balance parentheses so module compiles | Balanced parens (0.45) + compiles (0.45) |
| `import-fix` | Medium | Add missing `import json` | Import present (0.45) + compiles (0.45) |
| `docstring-stub` | Hard | Add docstring + return body to function | Docstring >12 chars (0.45) + return stmt (0.45) + compiles (0.05) |

## Setup

```bash
pip install -r requirements.txt
```

### Run the baseline inference (structured START/STEP/END output)

```bash
python inference.py
```

### Run the FastAPI server locally

```bash
uvicorn src.server:app --host 0.0.0.0 --port 7860
```

## Required Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | *(required)* | Hugging Face API token |

## Baseline Performance

Running `python inference.py` against `Qwen/Qwen2.5-72B-Instruct`:

| Task | Expected Reward |
|---|---|
| syntax-line | ≥ 0.90 |
| import-fix | ≥ 0.90 |
| docstring-stub | ≥ 0.90 |

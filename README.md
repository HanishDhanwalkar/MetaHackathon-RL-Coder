---
title: CRA-Environment
emoji: ⚡
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# CRA-Environment

Contextual Reinforcement Autocomplete — an RL-powered code completion environment with Knowledge Graph hints and deterministic per-task graders.

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**, Round 1.

## Tasks

| Task | Difficulty | Description | Grader |
|------|-----------|-------------|--------|
| `syntax-line` | Easy | Finish the line so the module is valid Python (balanced parentheses). | `grade_syntax_line` |
| `import-fix` | Medium | Add the missing import so the file runs. | `grade_import_fix` |
| `docstring-stub` | Hard | Give the function a docstring and a body with a return statement. | `grade_docstring_stub` |

All graders are deterministic and return a float score in `[0.0, 1.0]`.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset the environment |
| `/step` | POST | Take a step with an action |
| `/state` | GET | Get the current state |
| `/tasks` | GET | List all tasks with grader metadata |
| `/health` | GET | Health check |

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

## Docker

```bash
docker build -t cra-env .
docker run -p 7860:7860 cra-env
```

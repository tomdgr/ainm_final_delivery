---
name: Use uv not pip
description: User prefers uv package manager over pip for Python dependency management
type: feedback
---

Always use `uv` (not pip) for Python package management — `uv init`, `uv add`, `uv run`.

**Why:** User corrected when pip install was attempted; this is their preferred workflow.
**How to apply:** For any Python dependency install, use `uv add <package>`. For running scripts, use `uv run`.

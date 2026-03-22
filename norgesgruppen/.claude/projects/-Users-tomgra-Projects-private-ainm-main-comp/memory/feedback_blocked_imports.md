---
name: feedback_blocked_imports
description: CRITICAL - sandbox blocks os, sys, subprocess, pickle, etc. in run.py — auto-ban if violated
type: feedback
---

NEVER use these imports in generated run.py files for submissions:
os, sys, subprocess, socket, ctypes, pickle, marshal, yaml, requests, urllib, multiprocessing, threading, code, pty

Also blocked: eval(), exec(), compile(), __import__(), getattr() with dangerous names

Use `pathlib` instead of `os`. Use `json` instead of `yaml`.

**Why:** Team got AUTO-BANNED from competition for `import sys` in run.py. The sandbox security scanner auto-bans on any blocked import. This cost us submissions and time.

**How to apply:** Always check generated run.py code for blocked imports before building any submission ZIP. Add a validation step to all submission builders.

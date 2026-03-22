"""Test that all submission ZIPs pass competition rules.

Run: uv run python -m pytest test/test_submission_validation.py -v
"""
import zipfile
from pathlib import Path

import pytest

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def get_submission_zips():
    """Get all ZIP files in submissions/ (not in subdirs)."""
    return sorted(SUBMISSIONS_DIR.glob("*.zip"))


@pytest.fixture(params=get_submission_zips(), ids=lambda p: p.name)
def submission_zip(request):
    return request.param


def test_submissions_exist():
    zips = get_submission_zips()
    assert len(zips) > 0, "No submission ZIPs found"


def test_run_py_at_root(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        assert "run.py" in zf.namelist(), f"run.py not at ZIP root in {submission_zip.name}"


def test_max_weight_files(submission_zip):
    weight_exts = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
    with zipfile.ZipFile(submission_zip) as zf:
        weights = [n for n in zf.namelist() if Path(n).suffix in weight_exts]
        assert len(weights) <= 3, (
            f"{submission_zip.name} has {len(weights)} weight files (max 3): {weights}"
        )


def test_max_uncompressed_size(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        total = sum(i.file_size for i in zf.infolist())
        max_mb = 420
        assert total <= max_mb * 1024 * 1024, (
            f"{submission_zip.name} is {total / 1024 / 1024:.1f} MB uncompressed (max {max_mb})"
        )


def test_max_python_files(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        py_files = [n for n in zf.namelist() if n.endswith(".py")]
        assert len(py_files) <= 10, (
            f"{submission_zip.name} has {len(py_files)} Python files (max 10)"
        )


def test_max_total_files(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        assert len(zf.namelist()) <= 1000, (
            f"{submission_zip.name} has {len(zf.namelist())} files (max 1000)"
        )


def test_no_blocked_imports(submission_zip):
    blocked = {
        "os", "sys", "subprocess", "socket", "ctypes",
        "builtins", "importlib",
        "pickle", "marshal", "shelve", "shutil",
        "yaml",
        "requests", "urllib", "http",
        "multiprocessing", "threading", "signal",
        "gc", "code", "codeop", "pty",
    }
    import ast
    with zipfile.ZipFile(submission_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            code = zf.read(name).decode("utf-8", errors="replace")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        assert module not in blocked, (
                            f"{submission_zip.name}/{name}:{node.lineno} — "
                            f"import {alias.name} is BLOCKED (auto-ban!)"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    assert module not in blocked, (
                        f"{submission_zip.name}/{name}:{node.lineno} — "
                        f"from {node.module} is BLOCKED (auto-ban!)"
                    )


def test_no_blocked_calls(submission_zip):
    blocked_calls = {"eval", "exec", "compile", "__import__"}
    import ast
    with zipfile.ZipFile(submission_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            code = zf.read(name).decode("utf-8", errors="replace")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in blocked_calls, (
                        f"{submission_zip.name}/{name}:{node.lineno} — "
                        f"{node.func.id}() is BLOCKED"
                    )


def test_no_path_traversal(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        for name in zf.namelist():
            assert ".." not in name, f"Path traversal in {submission_zip.name}: {name}"
            assert not name.startswith("/"), f"Absolute path in {submission_zip.name}: {name}"


def test_allowed_extensions_only(submission_zip):
    allowed = {".py", ".json", ".yaml", ".yml", ".cfg", ".pt", ".pth", ".onnx", ".safetensors", ".npy", ""}
    with zipfile.ZipFile(submission_zip) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            ext = Path(name).suffix
            assert ext in allowed, (
                f"{submission_zip.name} contains disallowed file type: {name}"
            )


def test_run_py_has_input_output_args(submission_zip):
    with zipfile.ZipFile(submission_zip) as zf:
        if "run.py" not in zf.namelist():
            pytest.skip("No run.py")
        code = zf.read("run.py").decode("utf-8", errors="replace")
        assert "--input" in code, f"{submission_zip.name}/run.py missing --input arg"
        assert "--output" in code, f"{submission_zip.name}/run.py missing --output arg"

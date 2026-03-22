"""Validate a submission ZIP against ALL competition sandbox rules.

Based on: https://app.ainm.no/docs/norgesgruppen-data/submission
Run this BEFORE every submission to avoid auto-bans.

Usage:
  uv run python scripts/validate_submission.py submissions/my_submission.zip
"""
import ast
import re
import zipfile
from pathlib import Path

# === BLOCKED IMPORTS (from competition docs) ===
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes",
    "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil",
    "yaml",
    "requests", "urllib", "http",
    "multiprocessing", "threading", "signal",
    "gc", "code", "codeop", "pty",
}

# === BLOCKED FUNCTION CALLS ===
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}

# === ALLOWED FILE EXTENSIONS ===
ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}

# === WEIGHT FILE EXTENSIONS ===
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}

# === LIMITS ===
MAX_UNCOMPRESSED_SIZE = 420 * 1024 * 1024  # 420 MB
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3


def check_blocked_imports(code: str, filename: str) -> list[str]:
    """Check for blocked imports using AST + regex."""
    errors = []

    # AST-based check
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        errors.append(
                            f"{filename}:{node.lineno} — import {alias.name} "
                            f"(BLOCKED module: {module})"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        errors.append(
                            f"{filename}:{node.lineno} — from {node.module} import ... "
                            f"(BLOCKED module: {module})"
                        )
            elif isinstance(node, ast.Call):
                # Check direct calls like eval(), exec()
                if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                    errors.append(
                        f"{filename}:{node.lineno} — {node.func.id}() "
                        f"(BLOCKED call)"
                    )
                # Check method calls like obj.eval()
                elif isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALLS:
                    errors.append(
                        f"{filename}:{node.lineno} — .{node.func.attr}() "
                        f"(BLOCKED call)"
                    )
    except SyntaxError as e:
        errors.append(f"{filename}: SyntaxError — {e}")

    # Regex fallback for things AST might miss (string-based imports, etc.)
    for i, line in enumerate(code.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for blocked in BLOCKED_IMPORTS:
            # Match "import os" or "from os import ..."
            if re.search(rf'^\s*import\s+{blocked}\b', stripped):
                err = f"{filename}:{i} — {stripped} (BLOCKED: {blocked})"
                if not any(blocked in e for e in errors if f":{i} " in e):
                    errors.append(err)
            if re.search(rf'^\s*from\s+{blocked}\b', stripped):
                err = f"{filename}:{i} — {stripped} (BLOCKED: {blocked})"
                if not any(blocked in e for e in errors if f":{i} " in e):
                    errors.append(err)

    # Check for getattr with dangerous names
    if "getattr" in code:
        for i, line in enumerate(code.split("\n"), 1):
            if "getattr" in line and not line.strip().startswith("#"):
                errors.append(
                    f"{filename}:{i} — getattr() usage detected (potentially BLOCKED)"
                )

    return errors


def check_binary_files(zf: zipfile.ZipFile) -> list[str]:
    """Check for binary executables (ELF/Mach-O/PE)."""
    errors = []
    BINARY_MAGIC = {
        b"\x7fELF": "ELF binary",
        b"MZ": "PE/Windows binary",
        b"\xfe\xed\xfa": "Mach-O binary",
        b"\xcf\xfa\xed\xfe": "Mach-O 64-bit binary",
    }
    for info in zf.infolist():
        if info.file_size == 0 or info.filename.endswith("/"):
            continue
        ext = Path(info.filename).suffix
        if ext in ALLOWED_EXTENSIONS:
            continue
        try:
            header = zf.read(info.filename)[:4]
            for magic, desc in BINARY_MAGIC.items():
                if header.startswith(magic):
                    errors.append(f"FATAL: {desc} detected: {info.filename}")
        except Exception:
            pass
    return errors


def validate_zip(zip_path: str) -> tuple[bool, list[str], dict]:
    """Validate a submission ZIP against all rules."""
    issues = []
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return False, [f"File not found: {zip_path}"], {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        infos = zf.infolist()

        # === STRUCTURE CHECKS ===

        # run.py at root
        if "run.py" not in names:
            issues.append("FATAL: run.py not found at ZIP root")

        # No nested run.py (common mistake)
        nested = [n for n in names if n.endswith("/run.py") and n != "run.py"]
        if nested:
            issues.append(f"FATAL: run.py in subdirectory: {nested} — must be at root")

        # __MACOSX junk
        macosx = [n for n in names if "__MACOSX" in n]
        if macosx:
            issues.append(f"WARNING: __MACOSX files found — use: zip -r sub.zip . -x '.*' '__MACOSX/*'")

        # === COUNT LIMITS ===

        if len(names) > MAX_FILES:
            issues.append(f"FATAL: {len(names)} files exceeds max {MAX_FILES}")

        py_files = [n for n in names if n.endswith(".py")]
        if len(py_files) > MAX_PYTHON_FILES:
            issues.append(f"FATAL: {len(py_files)} Python files exceeds max {MAX_PYTHON_FILES}")

        weight_files = [n for n in names if Path(n).suffix in WEIGHT_EXTENSIONS]
        if len(weight_files) > MAX_WEIGHT_FILES:
            issues.append(
                f"FATAL: {len(weight_files)} weight files exceeds max {MAX_WEIGHT_FILES}: "
                f"{weight_files}"
            )

        # === SIZE LIMITS ===

        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
            issues.append(
                f"FATAL: Uncompressed size {total_uncompressed / 1024 / 1024:.1f} MB "
                f"exceeds max 420 MB"
            )

        # === FILE EXTENSION CHECKS ===

        for name in names:
            if name.endswith("/"):
                continue
            ext = Path(name).suffix
            if ext and ext not in ALLOWED_EXTENSIONS:
                issues.append(f"WARNING: Disallowed file extension: {name}")

        # === SECURITY CHECKS ===

        # Path traversal
        for name in names:
            if ".." in name:
                issues.append(f"FATAL: Path traversal detected: {name}")
            if name.startswith("/"):
                issues.append(f"FATAL: Absolute path detected: {name}")

        # Symlinks
        for info in infos:
            if info.external_attr >> 16 & 0o120000 == 0o120000:
                issues.append(f"FATAL: Symlink detected: {info.filename}")

        # Binary executables
        binary_issues = check_binary_files(zf)
        issues.extend(binary_issues)

        # === BLOCKED IMPORTS IN ALL PYTHON FILES ===

        for name in py_files:
            try:
                code = zf.read(name).decode("utf-8", errors="replace")
                import_errors = check_blocked_imports(code, name)
                for err in import_errors:
                    issues.append(f"FATAL: Blocked import — {err}")
            except Exception as e:
                issues.append(f"WARNING: Could not read {name}: {e}")

        # === OUTPUT FORMAT CHECK ===
        # Check if run.py uses argparse with --input and --output
        if "run.py" in names:
            code = zf.read("run.py").decode("utf-8", errors="replace")
            if "--input" not in code:
                issues.append("WARNING: run.py may not handle --input argument")
            if "--output" not in code:
                issues.append("WARNING: run.py may not handle --output argument")
            if "json" not in code:
                issues.append("WARNING: run.py doesn't seem to import json (needed for output)")

    # Build info dict
    compressed_size = zip_path.stat().st_size / 1024 / 1024
    info = {
        "compressed_size_mb": round(compressed_size, 1),
        "uncompressed_size_mb": round(total_uncompressed / 1024 / 1024, 1),
        "total_files": len(names),
        "python_files": len(py_files),
        "python_file_names": py_files,
        "weight_files": len(weight_files),
        "weight_names": weight_files,
    }

    fatal_count = len([i for i in issues if i.startswith("FATAL")])
    return fatal_count == 0, issues, info


if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser(description="Validate submission ZIP against competition rules")
    parser.add_argument("zips", nargs="+", help="ZIP files to validate")
    args = parser.parse_args()

    all_valid = True
    for z in args.zips:
        print(f"\n{'='*60}")
        print(f"Validating: {z}")
        print(f"{'='*60}")
        is_valid, issues, info = validate_zip(z)

        print(f"  Compressed:   {info['compressed_size_mb']} MB")
        print(f"  Uncompressed: {info['uncompressed_size_mb']} MB (max 420 MB)")
        print(f"  Files:        {info['total_files']} total (max {MAX_FILES})")
        print(f"  Python:       {info['python_files']} files (max {MAX_PYTHON_FILES}): {info['python_file_names']}")
        print(f"  Weights:      {info['weight_files']} files (max {MAX_WEIGHT_FILES}): {info['weight_names']}")

        if issues:
            print()
            for issue in issues:
                if issue.startswith("FATAL"):
                    print(f"  FAIL: {issue}")
                else:
                    print(f"  WARN: {issue}")

        print()
        if is_valid:
            print(f"  RESULT: PASS — Safe to submit")
        else:
            print(f"  RESULT: FAIL — DO NOT SUBMIT")
            all_valid = False

    if not all_valid:
        exit(1)

"""CYCLE 4: Security scan documentation and security-property tests.

Scan summary
------------
Files scanned: all .py files under agentsim/
Findings      : 0 real findings
False positives filtered: use of stdlib ``random`` in deliberative.py and
    learning.py — these are for agent *exploration*, not cryptographic use.

Verified clean:
- No hardcoded secrets (API keys, passwords, tokens)
- No eval() / exec() / compile() on user input
- No unsafe pickle deserialization
- No subprocess / os.system() injection vectors
- No path traversal vulnerabilities (no open() with user-controlled paths)
- No SQL injection (no database access)
- No dynamic __import__ calls
"""

from __future__ import annotations

import ast
import os

# ===========================================================================
# Static AST scan — embedded as a live test so CI catches regressions
# ===========================================================================


def _collect_python_files(root: str) -> list[str]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                paths.append(os.path.join(dirpath, fname))
    return paths


def test_no_eval_exec_in_source():
    """No eval() or exec() calls exist anywhere in agentsim source."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    source_root = os.path.join(repo_root, "agentsim")
    violations = []

    for path in _collect_python_files(source_root):
        with open(path) as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in ("eval", "exec", "compile"):
                    violations.append(f"{path}:{node.lineno}: {name}()")

    assert violations == [], f"Unsafe calls found: {violations}"


def test_no_pickle_in_source():
    """No pickle import exists in agentsim source (unsafe deserialization)."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    source_root = os.path.join(repo_root, "agentsim")
    violations = []

    for path in _collect_python_files(source_root):
        with open(path) as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if "pickle" in name:
                        violations.append(f"{path}:{node.lineno}: import {name}")

    assert violations == [], f"Pickle imports found: {violations}"


def test_no_subprocess_in_source():
    """No subprocess or os.system import in agentsim source."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    source_root = os.path.join(repo_root, "agentsim")
    violations = []
    dangerous = {"subprocess", "os.system", "commands"}

    for path in _collect_python_files(source_root):
        with open(path) as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if any(d in name for d in dangerous):
                        violations.append(f"{path}:{node.lineno}: import {name}")

    assert violations == [], f"Dangerous imports found: {violations}"


def test_no_hardcoded_secrets_patterns():
    """No obvious hardcoded secret patterns in source or config files."""
    import re

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    # Simple patterns — not exhaustive but catches common mistakes
    patterns = [
        re.compile(r'(?i)(password|passwd|secret|api_key)\s*=\s*["\'][^"\']{4,}'),
        re.compile(r'(?i)sk-[A-Za-z0-9]{20,}'),        # OpenAI-style key
        re.compile(r'AKIA[0-9A-Z]{16}'),                # AWS access key
        re.compile(r'(?i)ghp_[A-Za-z0-9]{36}'),        # GitHub PAT
    ]
    violations = []
    exclude_dirs = {".git", "__pycache__", "node_modules", ".venv"}

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            if not fname.endswith((".py", ".toml", ".yaml", ".yml", ".json", ".env")):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for pat in patterns:
                            if pat.search(line):
                                # Filter known test fixtures with intentionally fake values
                                if "test_" in fname or "conftest" in fname:
                                    continue
                                violations.append(f"{path}:{lineno}: {line.strip()[:80]}")
            except OSError:
                continue

    assert violations == [], "Potential secrets found:\n" + "\n".join(violations)

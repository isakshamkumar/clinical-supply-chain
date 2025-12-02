#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT  # Script is already in the project directory

# Optional: ignore some patterns
IGNORE_DIRS = {".git", "__pycache__", ".venv", ".idea", ".pytest_cache"}
IGNORE_EXTS = {".pyc", ".pyo", ".DS_Store"}

def should_skip(path: Path) -> bool:
    parts = set(p.name for p in path.parents)
    if parts & IGNORE_DIRS:
        return True
    if path.is_file() and path.suffix in IGNORE_EXTS:
        return True
    return False

chunks = []

for path in sorted(PROJECT_DIR.rglob("*")):
    if not path.is_file():
        continue
    if should_skip(path):
        continue

    rel_path = path.relative_to(PROJECT_DIR)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip non-text/binary files
        continue

    chunks.append(f"===== {rel_path} =====\n{content}\n")

full_text = "\n".join(chunks)

if not full_text:
    print("ERROR: No files found! Check that the script is in the correct directory.")
    exit(1)

# macOS clipboard
subprocess.run(["pbcopy"], input=full_text.encode("utf-8"), check=True)

print(f"✅ Copied {len(chunks)} files ({len(full_text)} characters) to clipboard.")
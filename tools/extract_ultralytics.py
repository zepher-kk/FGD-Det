#!/usr/bin/env python3






















from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List


EXCLUDE_DIR_NAMES = {"extra_modules", "backbone", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def path_contains_any_dir(path: Path, names: Iterable[str]) -> bool:

    s = set(names)
    return any(part in s for part in path.parts)


def copy_ultralytics_tree(src_pkg: Path, dst_pkg: Path, verbose: bool = True) -> None:

    if dst_pkg.exists():
        raise FileExistsError(f"Destination already exists: {dst_pkg}")

    for root, dirs, files in os.walk(src_pkg):
        root_path = Path(root)


        pruned: List[str] = []
        for d in list(dirs):
            dpath = root_path / d
            if d in EXCLUDE_DIR_NAMES or path_contains_any_dir(dpath, EXCLUDE_DIR_NAMES):
                dirs.remove(d)
                pruned.append(d)
        if verbose and pruned:
            print(f"[exclude] {root_path}: dirs -> {', '.join(pruned)}")


        rel = root_path.relative_to(src_pkg)
        dst_dir = dst_pkg / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if any(f.endswith(suf) for suf in EXCLUDE_FILE_SUFFIXES):
                if verbose:
                    print(f"[skip]    {root_path / f}")
                continue
            if f in {".DS_Store"}:
                continue
            src_file = root_path / f
            dst_file = dst_dir / f
            shutil.copy2(src_file, dst_file)
            if verbose:
                print(f"[copy]    {src_file} -> {dst_file}")


def _comment_lines_inplace(file_path: Path, line_indices: Iterable[int], prefix: str = "# ") -> int:

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    count = 0
    idx_set = set(line_indices)
    for i in sorted(idx_set):
        if i < 0 or i >= len(lines):
            continue
        line = lines[i]

        if line.lstrip().startswith("#"):
            continue

        leading_ws = re.match(r"^\s*", line).group(0)
        lines[i] = f"{leading_ws}{prefix}{line[len(leading_ws):]}"
        count += 1
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def comment_imports_tasks(file_path: Path, verbose: bool = True) -> int:

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    to_comment: List[int] = []

    pat = re.compile(r"^\s*from\s+ultralytics\.nn\.(?:extra_modules|backbone)\b")
    for i, line in enumerate(lines):
        if pat.search(line):
            to_comment.append(i)

    if not to_comment:
        return 0

    count = _comment_lines_inplace(file_path, to_comment)
    if verbose:
        print(f"[rewrite] {file_path}: commented {count} import lines (extra_modules/backbone)")
    return count


def comment_nnexpend_blocks(file_path: Path, verbose: bool = True) -> int:












    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    to_comment: List[int] = []

    marker_regex = re.compile(r"^(?P<indent>\s*)#\s*NNexpend\b")
    else_regex_by_indent_cache: dict[str, re.Pattern[str]] = {}
    mm_anchor_regex = re.compile(r"^\s*#\s*=+\s*MULTIMODAL\s+EXTENSION\s+START")

    i = 0
    n = len(lines)
    while i < n:
        m = marker_regex.match(lines[i])
        if not m:
            i += 1
            continue
        indent = m.group("indent")

        else_pat = else_regex_by_indent_cache.get(indent)
        if else_pat is None:
            else_pat = re.compile(rf"^{re.escape(indent)}else\s*:\s*$")
            else_regex_by_indent_cache[indent] = else_pat


        j = i + 1
        while j < n:
            if marker_regex.match(lines[j]):
                break
            if else_pat.match(lines[j]):
                break
            if mm_anchor_regex.match(lines[j]):
                break
            j += 1


        to_comment.extend(range(i + 1, j))
        i = j

    if not to_comment:
        return 0

    count = _comment_lines_inplace(file_path, to_comment)
    if verbose:
        print(f"[rewrite] {file_path}: commented {count} NNexpend lines in wrapped regions")
    return count


def main():
    ap = argparse.ArgumentParser(description="Extract ultralytics subset with exclusions and code tweaks.")
    ap.add_argument("--src", type=Path, default=Path("."), help="Project root (contains the ultralytics/ folder)")
    ap.add_argument("--dst", type=Path, required=True, help="Destination directory for extracted package")
    ap.add_argument("--package", type=str, default="ultralytics", help="Top-level package name to extract")
    ap.add_argument("--no-comment-imports", action="store_true", help="Do not comment extra_modules/backbone imports in tasks.py")
    ap.add_argument("--no-comment-nnexpend", action="store_true", help="Do not comment NNexpend-wrapped blocks in tasks.py")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    src_pkg = (args.src / args.package).resolve()
    if not src_pkg.exists() or not src_pkg.is_dir():
        raise FileNotFoundError(f"Package directory not found: {src_pkg}")

    dst_pkg = (args.dst / args.package).resolve()
    if args.verbose:
        print(f"[start]  extract from {src_pkg} -> {dst_pkg}")

    copy_ultralytics_tree(src_pkg, dst_pkg, verbose=args.verbose)


    tasks_py = dst_pkg / "nn" / "tasks.py"
    if tasks_py.exists():
        if not args.no_comment_imports:
            comment_imports_tasks(tasks_py, verbose=args.verbose)
        if not args.no_comment_nnexpend:
            comment_nnexpend_blocks(tasks_py, verbose=args.verbose)
    else:
        if args.verbose:
            print(f"[warn]   tasks.py not found at {tasks_py}, skipping rewrite")


    warn_patterns = [
        re.compile(r"ultralytics\.nn\.extra_modules"),
        re.compile(r"ultralytics\.nn\.backbone"),
    ]
    remaining_refs = []
    for root, _, files in os.walk(dst_pkg):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = Path(root) / f
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat in warn_patterns:
                if pat.search(content):
                    remaining_refs.append(str(p))
                    break
    if remaining_refs:
        print("[warn]   Remaining references to excluded modules detected:")
        for p in sorted(set(remaining_refs)):
            print(f"          - {p}")
    else:
        if args.verbose:
            print("[ok]     No remaining references to extra_modules/backbone found in destination")

    if args.verbose:
        print("[done]   extraction complete")


if __name__ == "__main__":
    main()


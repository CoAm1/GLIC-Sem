#!/usr/bin/env python3
"""Reject model, dataset, result, secret, binary, or oversized Git payloads."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath


BANNED_SUFFIXES = {
    ".bag", ".bin", ".ckpt", ".db3", ".engine", ".mcap", ".npy", ".npz",
    ".onnx", ".pcd", ".ply", ".pt", ".pth", ".safetensors", ".tar", ".tgz",
    ".zip", ".7z",
}
BANNED_DIRECTORY_PARTS = {
    "artifacts", "build", "ckpt", "data", "datasets", "devel", "logs",
    "models", "output", "outputs", "result", "results", "runs", "weights",
}
SENSITIVE_NAMES = {
    ".env", "credentials", "credentials.json", "id_dsa", "id_ed25519",
    "id_ecdsa", "id_rsa", "known_hosts", "yuhet",
}
TEXT_SUFFIXES = {
    "", ".c", ".cc", ".cmake", ".cpp", ".cu", ".cuh", ".csv", ".h", ".hpp",
    ".json", ".md", ".py", ".sh", ".txt", ".xml", ".yaml", ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-mib", type=float, default=10.0)
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Audit explicit repository-relative paths instead of staged files.",
    )
    return parser.parse_args()


def staged_paths(repo: Path) -> list[str]:
    process = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        value.decode("utf-8", errors="strict")
        for value in process.stdout.split(b"\0") if value
    ]


def audit_path(repo: Path, relative: str, maximum_bytes: int) -> list[str]:
    normalized = PurePosixPath(relative.replace("\\", "/"))
    errors = []
    if normalized.is_absolute() or ".." in normalized.parts:
        return [f"unsafe repository path: {relative}"]
    lower_parts = {part.lower() for part in normalized.parts[:-1]}
    if lower_parts & BANNED_DIRECTORY_PARTS:
        errors.append(f"generated/data directory is forbidden: {relative}")
    filename = normalized.name.lower()
    if filename in SENSITIVE_NAMES or filename.startswith(".env."):
        errors.append(f"possible credential file is forbidden: {relative}")
    suffix = Path(filename).suffix.lower()
    if suffix in BANNED_SUFFIXES:
        errors.append(f"model/data/binary extension is forbidden: {relative}")
    absolute = repo / Path(*normalized.parts)
    if not absolute.exists():
        errors.append(f"staged path does not exist in worktree: {relative}")
        return errors
    if not absolute.is_file():
        return errors
    size = absolute.stat().st_size
    if size > maximum_bytes:
        errors.append(
            f"file exceeds {maximum_bytes / (1024 * 1024):g} MiB: "
            f"{relative} ({size} bytes)"
        )
    if suffix not in TEXT_SUFFIXES and size:
        with absolute.open("rb") as source:
            prefix = source.read(8192)
        if b"\0" in prefix:
            errors.append(f"unapproved binary payload: {relative}")
    return errors


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    maximum_bytes = int(args.max_mib * 1024 * 1024)
    if maximum_bytes < 1:
        raise ValueError("max-mib must be positive")
    paths = args.paths if args.paths is not None else staged_paths(repo)
    errors = []
    for relative in paths:
        errors.extend(audit_path(repo, relative, maximum_bytes))
    if errors:
        for error in errors:
            print(f"[GIT-PAYLOAD-FAIL] {error}", file=sys.stderr)
        return 1
    print(f"[GIT-PAYLOAD-PASS] audited {len(paths)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

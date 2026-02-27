#!/usr/bin/env python3
"""Summarize contributed lines per author for src/harpy Python files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass


COMMIT_MARKER = "@@COMMIT@@"
CO_AUTHOR_RE = re.compile(r"^Co-authored-by:\s*([^<]+)<[^>]+>\s*$", re.IGNORECASE | re.MULTILINE)
AUTHOR_ALIASES = {
    "arned": "Arne Defauw",
    "arne defauw": "Arne Defauw",
    "arnedefauw": "Arne Defauw",
}


@dataclass
class LineStats:
    added: float = 0
    deleted: float = 0

    @property
    def net(self) -> float:
        return self.added - self.deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show number of lines contributed per author in Python files, "
            "split by core and _tests."
        )
    )
    parser.add_argument(
        "--root",
        default="src/harpy",
        help="Root path to analyze (default: src/harpy).",
    )
    parser.add_argument(
        "--no-coauthors",
        action="store_true",
        help="Do not include Co-authored-by trailers in attribution.",
    )
    parser.add_argument(
        "--mode",
        choices=["blame", "numstat"],
        default="blame",
        help=(
            "Attribution mode: "
            "'blame' counts current lines by author (default), "
            "'numstat' counts historical added/deleted lines from commits."
        ),
    )
    return parser.parse_args()


def gather_coauthors(root: str) -> dict[str, list[str]]:
    cmd = [
        "git",
        "log",
        "--format=%x1e%H%x1f%B",
        "--",
        root,
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    coauthors_by_commit: dict[str, list[str]] = {}
    for raw_record in proc.stdout.split("\x1e"):
        record = raw_record.strip()
        if not record:
            continue
        if "\x1f" not in record:
            continue
        commit_hash, body = record.split("\x1f", maxsplit=1)
        names = [m.strip() for m in CO_AUTHOR_RE.findall(body) if m.strip()]
        # Keep order and deduplicate.
        seen = set()
        deduped = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        coauthors_by_commit[commit_hash] = deduped
    return coauthors_by_commit


def list_python_files(root: str) -> list[str]:
    cmd = ["git", "ls-files", "--", root]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip().endswith(".py")]
    return files


def normalize_author(author: str) -> str:
    key = author.strip().lower()
    return AUTHOR_ALIASES.get(key, author.strip() or "UNKNOWN")


def gather_blame_stats(root: str) -> dict[tuple[str, str], int]:
    stats: dict[tuple[str, str], int] = defaultdict(int)
    files = list_python_files(root)

    for path in files:
        category = "_tests" if "/_tests/" in path else "core"
        cmd = ["git", "blame", "--line-porcelain", "--", path]
        try:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            print(exc.stderr.strip() or str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

        for line in proc.stdout.splitlines():
            if not line.startswith("author "):
                continue
            author = normalize_author(line[len("author ") :])
            stats[(category, author)] += 1

    return stats


def gather_stats(root: str, include_coauthors: bool) -> dict[tuple[str, str], LineStats]:
    coauthors_by_commit = gather_coauthors(root) if include_coauthors else {}
    cmd = [
        "git",
        "log",
        "--numstat",
        f"--format={COMMIT_MARKER}%H%x09%aN",
        "--",
        root,
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    stats: dict[tuple[str, str], LineStats] = defaultdict(LineStats)
    current_author = "UNKNOWN"
    current_commit = ""

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(COMMIT_MARKER):
            payload = line[len(COMMIT_MARKER) :]
            commit_and_author = payload.split("\t", maxsplit=1)
            current_commit = commit_and_author[0].strip()
            if len(commit_and_author) > 1:
                current_author = commit_and_author[1].strip() or "UNKNOWN"
            else:
                current_author = "UNKNOWN"
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added_str, deleted_str, path = parts
        if not path.endswith(".py"):
            continue

        category = "_tests" if "/_tests/" in path else "core"
        added = int(added_str) if added_str.isdigit() else 0
        deleted = int(deleted_str) if deleted_str.isdigit() else 0

        recipients = [current_author]
        if include_coauthors:
            for coauthor in coauthors_by_commit.get(current_commit, []):
                if coauthor not in recipients:
                    recipients.append(coauthor)

        for recipient in recipients:
            key = (category, normalize_author(recipient))
            stats[key].added += added
            stats[key].deleted += deleted

    return stats


def format_num(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def print_table(title: str, rows: list[tuple[str, float, float, float]]) -> None:
    print(title)
    print("-" * len(title))
    print(f"{'Author':30} {'Added':>10} {'Deleted':>10} {'Net':>10}")
    print("-" * 64)
    for author, added, deleted, net in rows:
        print(f"{author:30} {format_num(added):>10} {format_num(deleted):>10} {format_num(net):>10}")
    print()


def print_blame_table(title: str, rows: list[tuple[str, int]]) -> None:
    print(title)
    print("-" * len(title))
    print(f"{'Author':30} {'Lines':>10}")
    print("-" * 42)
    for author, lines in rows:
        print(f"{author:30} {lines:10d}")
    print()


def main() -> None:
    args = parse_args()
    if args.mode == "blame":
        blame_stats = gather_blame_stats(args.root)
        if not blame_stats:
            print("No matching lines found.")
            return

        categories = ["core", "_tests"]
        for category in categories:
            rows = [
                (author, lines)
                for (cat, author), lines in blame_stats.items()
                if cat == category
            ]
            rows.sort(key=lambda row: (row[1], row[0].lower()), reverse=True)
            print_blame_table(f"{category} (.py files, current lines)", rows)

        totals = Counter()
        for (_, author), lines in blame_stats.items():
            totals[author] += lines
        total_rows = sorted(
            totals.items(),
            key=lambda row: (row[1], row[0].lower()),
            reverse=True,
        )
        print_blame_table("total (.py files, current lines)", total_rows)
        return

    stats = gather_stats(args.root, include_coauthors=not args.no_coauthors)

    if not stats:
        print("No matching commits found.")
        return

    categories = ["core", "_tests"]
    for category in categories:
        category_rows = []
        for (cat, author), line_stats in stats.items():
            if cat != category:
                continue
            category_rows.append((author, line_stats.added, line_stats.deleted, line_stats.net))
        category_rows.sort(key=lambda row: (row[1], row[3], row[0].lower()), reverse=True)
        suffix = " (includes co-authors)" if not args.no_coauthors else ""
        print_table(f"{category} (.py files){suffix}", category_rows)

    totals: dict[str, LineStats] = defaultdict(LineStats)
    for (_, author), line_stats in stats.items():
        totals[author].added += line_stats.added
        totals[author].deleted += line_stats.deleted

    total_rows = [
        (author, s.added, s.deleted, s.net)
        for author, s in totals.items()
    ]
    total_rows.sort(key=lambda row: (row[1], row[3], row[0].lower()), reverse=True)
    suffix = " (includes co-authors)" if not args.no_coauthors else ""
    print_table(f"total (.py files){suffix}", total_rows)


if __name__ == "__main__":
    main()

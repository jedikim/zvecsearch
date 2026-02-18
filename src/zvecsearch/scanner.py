from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    mtime: float
    size: int


def scan_paths(
    paths: list[str | Path],
    extensions: tuple[str, ...] = (".md", ".markdown"),
    ignore_hidden: bool = True,
) -> list[ScannedFile]:
    seen: set[str] = set()
    results: list[ScannedFile] = []

    for p in paths:
        p = Path(p)
        if p.is_file():
            _maybe_add(p, extensions, seen, results, ignore_hidden)
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                if ignore_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in sorted(files):
                    fp = Path(root) / fname
                    _maybe_add(fp, extensions, seen, results, ignore_hidden)

    results.sort(key=lambda f: f.path)
    return results


def _maybe_add(
    fp: Path,
    extensions: tuple[str, ...],
    seen: set[str],
    results: list[ScannedFile],
    ignore_hidden: bool,
) -> None:
    if ignore_hidden and fp.name.startswith("."):
        return
    if fp.suffix.lower() not in extensions:
        return
    resolved = str(fp.resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    st = fp.stat()
    results.append(ScannedFile(path=fp.resolve(), mtime=st.st_mtime, size=st.st_size))

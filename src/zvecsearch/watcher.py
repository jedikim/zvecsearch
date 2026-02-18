from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class _MarkdownHandler(FileSystemEventHandler):
    def __init__(
        self,
        callback: Callable[[str, Path], None],
        extensions: tuple[str, ...] = (".md", ".markdown"),
        debounce_ms: int = 1500,
    ):
        self._callback = callback
        self._extensions = extensions
        self._debounce_s = debounce_ms / 1000.0
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule("deleted", event.src_path)

    def _schedule(self, event_type: str, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in self._extensions:
            return
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(
                self._debounce_s, self._fire, args=(event_type, path)
            )
            self._timers[path] = timer
            timer.start()

    def _fire(self, event_type: str, path: str) -> None:
        with self._lock:
            self._timers.pop(path, None)
        self._callback(event_type, Path(path))

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()


class FileWatcher:
    def __init__(
        self,
        paths: list[str | Path],
        callback: Callable[[str, Path], None],
        debounce_ms: int = 1500,
    ):
        self._paths = [Path(p) for p in paths]
        self._handler = _MarkdownHandler(callback, debounce_ms=debounce_ms)
        self._observer = Observer()

    def start(self) -> None:
        for p in self._paths:
            if p.is_dir():
                self._observer.schedule(self._handler, str(p), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join(timeout=2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

import pytest
import time
from pathlib import Path
from zvecsearch.watcher import FileWatcher


def test_watcher_start_stop(tmp_path):
    events = []
    def cb(event_type, path):
        events.append((event_type, path))

    watcher = FileWatcher([tmp_path], cb, debounce_ms=100)
    watcher.start()
    time.sleep(0.2)
    watcher.stop()


def test_watcher_detects_new_file(tmp_path):
    events = []
    def cb(event_type, path):
        events.append((event_type, path))

    watcher = FileWatcher([tmp_path], cb, debounce_ms=200)
    watcher.start()
    time.sleep(0.1)
    (tmp_path / "new.md").write_text("# New")
    time.sleep(0.5)
    watcher.stop()
    assert len(events) > 0


def test_watcher_ignores_non_markdown(tmp_path):
    events = []
    def cb(event_type, path):
        events.append((event_type, path))

    watcher = FileWatcher([tmp_path], cb, debounce_ms=100)
    watcher.start()
    time.sleep(0.1)
    (tmp_path / "test.txt").write_text("not markdown")
    time.sleep(0.3)
    watcher.stop()
    assert len(events) == 0

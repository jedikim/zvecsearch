from pathlib import Path
from zvecsearch.config import (
    ZvecSearchConfig,
    load_config_file, deep_merge, resolve_config,
    config_to_dict, get_config_value, save_config,
)


def test_default_config():
    cfg = ZvecSearchConfig()
    assert cfg.zvec.path == "~/.zvecsearch/db"
    assert cfg.zvec.collection == "zvecsearch_chunks"
    assert cfg.embedding.provider == "openai"
    assert cfg.chunking.max_chunk_size == 1500
    assert cfg.index.type == "hnsw"
    assert cfg.index.metric == "cosine"


def test_load_missing_file():
    assert load_config_file(Path("/nonexistent")) == {}


def test_load_toml_file(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('[zvec]\npath = "/custom/db"\n')
    d = load_config_file(f)
    assert d["zvec"]["path"] == "/custom/db"


def test_deep_merge_basic():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 99}}
    result = deep_merge(base, over)
    assert result == {"a": {"x": 1, "y": 99}, "b": 3}


def test_deep_merge_none_skipped():
    base = {"a": 1}
    over = {"a": None}
    result = deep_merge(base, over)
    assert result == {"a": 1}


def test_resolve_with_cli_overrides():
    cfg = resolve_config({"zvec": {"collection": "custom"}})
    assert cfg.zvec.collection == "custom"
    assert cfg.zvec.path == "~/.zvecsearch/db"


def test_config_to_dict():
    cfg = ZvecSearchConfig()
    d = config_to_dict(cfg)
    assert d["zvec"]["path"] == "~/.zvecsearch/db"
    assert d["index"]["type"] == "hnsw"


def test_get_config_value():
    cfg = ZvecSearchConfig()
    assert get_config_value("zvec.collection", cfg) == "zvecsearch_chunks"
    assert get_config_value("index.metric", cfg) == "cosine"


def test_save_and_load(tmp_path):
    d = {"zvec": {"path": "/tmp/test"}}
    f = tmp_path / "test.toml"
    save_config(d, f)
    loaded = load_config_file(f)
    assert loaded["zvec"]["path"] == "/tmp/test"

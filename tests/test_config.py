from pathlib import Path
from zvecsearch.config import (
    ZvecSearchConfig,
    ZvecConfig,
    EmbeddingConfig,
    SearchConfig,
    load_config_file, deep_merge, resolve_config,
    config_to_dict, get_config_value, save_config,
)


def test_default_config():
    cfg = ZvecSearchConfig()
    assert cfg.zvec.path == "~/.zvecsearch/db"
    assert cfg.zvec.collection == "zvecsearch_chunks"
    assert cfg.zvec.quantize_type == "int8"
    assert cfg.zvec.hnsw_m == 16
    assert cfg.zvec.hnsw_ef == 300
    assert cfg.zvec.read_only is False
    assert cfg.zvec.enable_mmap is True


def test_embedding_config_defaults():
    cfg = ZvecSearchConfig()
    assert cfg.embedding.provider == "openai"
    assert cfg.embedding.model == "text-embedding-3-small"


def test_search_config_defaults():
    cfg = ZvecSearchConfig()
    assert cfg.search.query_ef == 300
    assert cfg.search.reranker == "rrf"
    assert cfg.search.dense_weight == 1.0
    assert cfg.search.sparse_weight == 0.8


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


def test_resolve_config_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr("zvecsearch.config._GLOBAL_CFG", tmp_path / "global.toml")
    monkeypatch.setattr("zvecsearch.config._PROJECT_CFG", tmp_path / "project.toml")
    cfg = resolve_config()
    assert cfg.zvec.path == "~/.zvecsearch/db"
    assert cfg.zvec.quantize_type == "int8"
    assert cfg.embedding.model == "text-embedding-3-small"
    assert cfg.search.reranker == "rrf"


def test_resolve_config_cli_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr("zvecsearch.config._GLOBAL_CFG", tmp_path / "global.toml")
    monkeypatch.setattr("zvecsearch.config._PROJECT_CFG", tmp_path / "project.toml")
    cfg = resolve_config({
        "zvec": {"quantize_type": "float16"},
        "search": {"reranker": "weighted"},
    })
    assert cfg.zvec.quantize_type == "float16"
    assert cfg.search.reranker == "weighted"


def test_config_to_dict():
    cfg = ZvecSearchConfig()
    d = config_to_dict(cfg)
    assert d["zvec"]["path"] == "~/.zvecsearch/db"
    assert d["zvec"]["quantize_type"] == "int8"
    assert "search" in d
    assert d["search"]["reranker"] == "rrf"
    assert d["search"]["dense_weight"] == 1.0


def test_get_config_value():
    cfg = ZvecSearchConfig()
    assert get_config_value("zvec.collection", cfg) == "zvecsearch_chunks"
    assert get_config_value("zvec.quantize_type", cfg) == "int8"
    assert get_config_value("search.reranker", cfg) == "rrf"
    assert get_config_value("embedding.model", cfg) == "text-embedding-3-small"


def test_save_and_load(tmp_path):
    d = {"zvec": {"path": "/tmp/test"}}
    f = tmp_path / "test.toml"
    save_config(d, f)
    loaded = load_config_file(f)
    assert loaded["zvec"]["path"] == "/tmp/test"

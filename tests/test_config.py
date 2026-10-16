import importlib

import pytest

import src.config as config


def test_relative_paths_resolve_against_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "indexes/chroma")
    monkeypatch.setenv("BM25_INDEX_PATH", "indexes/bm25.pkl")
    cfg = importlib.reload(config)
    try:
        assert cfg.CHROMA_PERSIST_DIR == str(cfg.PROJECT_ROOT / "indexes" / "chroma")
        assert cfg.BM25_INDEX_PATH == str(cfg.PROJECT_ROOT / "indexes" / "bm25.json")  # legacy .pkl mapped
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_absolute_path_kept(tmp_path):
    assert config.resolve_path(tmp_path) == tmp_path


def test_fallback_models_parsed(monkeypatch):
    monkeypatch.setenv("FALLBACK_MODELS", " a, b ,,a ")
    cfg = importlib.reload(config)
    try:
        assert cfg.FALLBACK_MODELS == ["a", "b", "a"]  # dedup happens in llm_client
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_check_indexes_missing_is_actionable(tmp_path):
    with pytest.raises(config.IndexesMissingError, match="build_index"):
        config.check_indexes(str(tmp_path / "nochroma"), str(tmp_path / "no.pkl"))


def test_check_indexes_present(tmp_path):
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "x").write_text("1")
    bm25 = tmp_path / "bm25.pkl"
    bm25.write_text("1")
    config.check_indexes(str(chroma), str(bm25))

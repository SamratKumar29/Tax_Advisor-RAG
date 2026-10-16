import json

from src.ingest import download_sources as ds


def _registry(tmp_path, entries):
    p = tmp_path / "sources.json"
    p.write_text(json.dumps({"sources": entries}))
    return p


def test_status_missing_unhashed_ok_and_mismatch(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.pdf").write_bytes(b"hello")
    good = ds.sha256_of(raw / "a.pdf")
    assert ds.check_source({"path": "a.pdf", "sha256": None}, raw) == "ok_unhashed"
    assert ds.check_source({"path": "a.pdf", "sha256": good}, raw) == "ok"
    assert ds.check_source({"path": "a.pdf", "sha256": "0" * 64}, raw) == "hash_mismatch"
    assert ds.check_source({"path": "b.pdf", "sha256": None}, raw) == "missing"


def test_main_exit_code_and_record_hashes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.pdf").write_bytes(b"x")
    reg = _registry(tmp_path, [{"path": "a.pdf", "url": None, "sha256": None}])
    assert ds.main(["--record-hashes"], registry_path=reg, raw_dir=raw) == 0
    assert json.loads(reg.read_text())["sources"][0]["sha256"] == ds.sha256_of(raw / "a.pdf")

    (raw / "a.pdf").write_bytes(b"tampered")
    assert ds.main([], registry_path=reg, raw_dir=raw) == 1


def test_download_from_direct_url(tmp_path):
    src = tmp_path / "remote.pdf"
    src.write_bytes(b"pdf-bytes")
    raw = tmp_path / "raw"
    entry = {"path": "circulars/c.pdf", "url": src.as_uri(), "url_is_direct": True}
    assert ds.download(entry, raw)
    assert (raw / "circulars" / "c.pdf").read_bytes() == b"pdf-bytes"


def test_non_direct_url_is_not_downloaded(tmp_path):
    assert ds.download({"path": "x.pdf", "url": "https://example.org/", "url_is_direct": False},
                       tmp_path) is False


def test_repo_registry_is_valid():
    reg = ds.load_registry()
    paths = [e["path"] for e in reg["sources"]]
    assert len(paths) == len(set(paths))
    assert all({"path", "url", "url_is_direct", "source_type", "sha256"} <= set(e) for e in reg["sources"])

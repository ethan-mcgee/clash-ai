import os

from clashai import paths


def _tree(root):
    (root / "core/src/main").mkdir(parents=True)
    (root / "env/src/main").mkdir(parents=True)
    (root / "core/src/main/A.java").write_text("class A {}")
    (root / "env/src/main/B.java").write_text("class B {}")
    (root / "build.gradle.kts").write_text("plugins {}")


def test_the_engine_fingerprint_follows_content_not_file_times(tmp_path):
    _tree(tmp_path)
    first = paths.engine_sources_hash(tmp_path)
    a = tmp_path / "core/src/main/A.java"
    os.utime(a, (1, 1))  # a checkout touching a file changes nothing
    assert paths.engine_sources_hash(tmp_path) == first
    a.write_text("class A { int x; }")
    assert paths.engine_sources_hash(tmp_path) != first
    a.write_text("class A {}")
    assert paths.engine_sources_hash(tmp_path) == first
    (tmp_path / "env/src/main/C.java").write_text("class C {}")  # a new file counts
    assert paths.engine_sources_hash(tmp_path) != first


def test_the_built_engine_matches_its_sources():
    install = paths.install_dir()
    marker = install / ".clashai-sources"
    if not os.environ.get("CLASHAI_CRFORGE_INSTALL"):
        assert marker.read_text().strip() == paths.engine_sources_hash()

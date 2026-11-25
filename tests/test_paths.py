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


def test_the_cpu_budget_is_16_by_default_and_follows_the_setting(monkeypatch):
    total = os.cpu_count()
    monkeypatch.delenv("CLASHAI_MAX_THREADS", raising=False)
    assert paths.cpu_budget() == min(16, total)
    monkeypatch.setenv("CLASHAI_MAX_THREADS", "4")
    assert paths.cpu_budget() == 4 and paths.default_threads() == 2
    monkeypatch.setenv("CLASHAI_MAX_THREADS", "all")
    assert paths.cpu_budget() == total
    monkeypatch.setenv("CLASHAI_MAX_THREADS", "999")
    assert paths.cpu_budget() == total


def test_importing_clashai_confines_the_process_to_the_budget():
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system))
    assert bin(process.value).count("1") == paths.cpu_budget()

import pytest

from clashai import paths


def _engine_ready() -> bool:
    return (paths.install_dir() / "lib").is_dir() and paths.tables_dir().is_dir()


def pytest_collection_modifyitems(config, items):
    if _engine_ready():
        return
    skip = pytest.mark.skip(reason="crforge engine or game tables not built (docs/SETUP.md)")
    for item in items:
        if "engine" in item.keywords:
            item.add_marker(skip)

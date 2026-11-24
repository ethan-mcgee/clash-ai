"""Where the engine, game tables and data live, and which Java runs crforge.

Every location can be overridden with an environment variable, so nothing machine-specific is
hard-coded beyond the defaults:

- ``CLASHAI_DATA``: the data root (datasets, checkpoints, runs, the crforge cache)
- ``CLASHAI_CRFORGE_INSTALL``: crforge's installed env distribution
- ``CLASHAI_TABLES``: the game tables folder of :data:`DATA_VERSION`
- ``CLASHAI_JAVA_HOME``: the JDK (21) that runs crforge
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRFORGE = REPO / "third_party" / "crforge"
DATA_VERSION = "16.635.8"
"""The game data version the engine, tables and replays all match."""


def data_root() -> Path:
    return Path(os.environ.get("CLASHAI_DATA", "D:/ml-data/clash-ai"))


def runs_dir() -> Path:
    return data_root() / "runs"


def install_dir() -> Path:
    default = CRFORGE / "env" / "build" / "install" / "env"
    return Path(os.environ.get("CLASHAI_CRFORGE_INSTALL", default))


def tables_dir() -> Path:
    default = data_root() / "crforge-cache" / "tables" / DATA_VERSION
    return Path(os.environ.get("CLASHAI_TABLES", default))


def _java_major(home: Path) -> int | None:
    release = home / "release"
    if not release.is_file():
        return None
    for line in release.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("JAVA_VERSION="):
            return int(line.split("=", 1)[1].strip().strip('"').split(".")[0])
    return None


def ensure_java() -> Path:
    """Points ``JAVA_HOME`` at a JDK 21 before JPype starts the JVM, and returns it.

    crforge is built and tested on Java 21. This machine's user ``JAVA_HOME`` names Java 25, and
    JPype starts whatever ``JAVA_HOME`` names, so it is set here explicitly: ``CLASHAI_JAVA_HOME``
    when given, else ``JAVA_HOME`` when it is already 21, else the newest Temurin 21 installed.
    """
    candidates: list[Path] = []
    if os.environ.get("CLASHAI_JAVA_HOME"):
        candidates.append(Path(os.environ["CLASHAI_JAVA_HOME"]))
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]))
    adoptium = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Eclipse Adoptium"
    if adoptium.is_dir():
        candidates += sorted(adoptium.glob("jdk-21*"), reverse=True)
    for home in candidates:
        if _java_major(home) == 21:
            os.environ["JAVA_HOME"] = str(home)
            return home
    raise RuntimeError("no JDK 21 found: install Temurin 21 or set CLASHAI_JAVA_HOME")


def default_threads() -> int:
    """JVM threads for stepping battles: one per physical core.

    Battles stop speeding up past the physical cores (measured in docs/SETUP.md), so this halves
    the logical count. Read from the machine, never fixed, so it grows when the BIOS re-enables
    the 7900X's second core cluster (12 cores / 24 threads).
    """
    return max(1, (os.cpu_count() or 2) // 2)

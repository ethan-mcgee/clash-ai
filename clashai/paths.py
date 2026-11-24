"""Where the engine, game tables and data live, and which Java runs crforge.

Every location can be overridden with an environment variable, so nothing machine-specific is
hard-coded beyond the defaults:

- ``CLASHAI_DATA``: the data root (datasets, checkpoints, runs, the crforge cache)
- ``CLASHAI_CRFORGE_INSTALL``: crforge's installed env distribution (when set, used as is and
  never rebuilt)
- ``CLASHAI_NO_ENGINE_BUILD``: set to 1 to never rebuild the engine automatically
- ``CLASHAI_TABLES``: the game tables folder of :data:`DATA_VERSION`
- ``CLASHAI_JAVA_HOME``: the JDK (21) that runs crforge
"""

from __future__ import annotations

import hashlib
import os
import subprocess
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
    """crforge's installed engine, rebuilt first if its Java sources changed (see ensure_engine)."""
    if os.environ.get("CLASHAI_CRFORGE_INSTALL"):
        return Path(os.environ["CLASHAI_CRFORGE_INSTALL"])
    ensure_engine()
    return CRFORGE / "env" / "build" / "install" / "env"


# What the engine install is built from: the env module, the core it depends on, and the build files.
ENGINE_SOURCES = ("core/src/main", "env/src/main", "core/build.gradle.kts", "env/build.gradle.kts",
                  "build.gradle.kts", "settings.gradle.kts", "gradle.properties",
                  "gradle/libs.versions.toml")
_MARKER = ".clashai-sources"
_checked = False


def engine_sources_hash(root: Path = CRFORGE) -> str:
    """A hash of every engine source file's path and bytes (not times, so a checkout that only
    touches files does not count as a change)."""
    h = hashlib.sha256()
    for entry in ENGINE_SOURCES:
        p = root / entry
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.is_file():
                h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def ensure_engine(force: bool = False) -> None:
    """Rebuilds crforge's engine (``gradlew :env:installDist``) when its Java sources differ from
    the last build's: the Java side's equivalent of an editable install. Checked once a process.

    Gradle compiles only what changed. The build uses the JDK of :func:`ensure_java` and the Gradle
    home on the data drive, offline (its dependencies are cached by the first build).
    """
    global _checked
    if (_checked and not force) or os.environ.get("CLASHAI_NO_ENGINE_BUILD") == "1":
        return
    install = CRFORGE / "env" / "build" / "install" / "env"
    marker = install / _MARKER
    digest = engine_sources_hash()
    if not force and install.is_dir() and marker.is_file() and marker.read_text().strip() == digest:
        _checked = True
        return
    java = ensure_java()
    env = dict(os.environ, JAVA_HOME=str(java),
               GRADLE_USER_HOME=os.environ.get("GRADLE_USER_HOME",
                                               str(data_root() / "gradle-home")))
    gradlew = CRFORGE / ("gradlew.bat" if os.name == "nt" else "gradlew")
    print("crforge's Java sources changed since the engine was built: rebuilding "
          "(gradlew :env:installDist)", flush=True)
    done = subprocess.run([str(gradlew), "--no-daemon", "--offline", "-q", ":env:installDist"],
                          cwd=CRFORGE, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"crforge's engine build failed:\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}")
    marker.write_text(digest + "\n")
    _checked = True


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

    Hyperthreads add little until batches are large (docs/SETUP.md), and the Python and GPU side
    needs CPU too, so this halves the logical count: 12 on the 7900X. Pass ``threads`` explicitly
    to use all 24 for engine-only work such as replay conversion.
    """
    return max(1, (os.cpu_count() or 2) // 2)

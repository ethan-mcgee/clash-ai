# Machine setup

How this project's Windows 11 machine runs crforge and trains on it. crforge's own docs are in [third_party/crforge/docs](../third_party/crforge/docs); this file records the choices, commands and fixes specific to this setup.

## Hardware

- GPU: RTX 3090 Ti, 24 GB, NVIDIA driver 621.14
- CPU: Ryzen 9 7900X, 12 cores / 24 threads. **As of 2026-11-24 Windows sees only 6 cores / 12 threads**: one core cluster is disabled in the BIOS. The full 12 / 24 will be re-enabled soon; the throughput below was measured on 6 / 12 and will be re-measured then. Code sizes thread counts from the machine, never a hard-coded 12.
- RAM: 64 GB
- Disks: C: is nearly full, so everything large goes on **D:**

## Layout

| What | Where |
|---|---|
| This repo | OneDrive Desktop (small, synced) |
| crforge | `third_party/crforge`, vendored with `git subtree` from `c56c196` (2026-11-22); see [CRFORGE.md](CRFORGE.md) |
| Python venv | `D:\ml-data\clash-ai\venv` |
| Gradle home | `D:\ml-data\clash-ai\gradle-home` (about 220 MB of build tools) |
| Game data cache and tables | `D:\ml-data\clash-ai\crforge-cache` (`tables\16.635.8`) |
| Cache backup | `D:\ml-data\clash-ai\backups\crforge-cache-16.635.8-2026-11-24.zip` |
| Datasets, checkpoints, runs | `D:\ml-data\clash-ai\{datasets,checkpoints,runs}` |

None of the D: folders are in the repo.

## Tools

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12.6 | `C:\Python312` |
| PyTorch | 2.5.1, CUDA 12.4 build | From `https://download.pytorch.org/whl/cu124`; the PyPI wheel on Windows is CPU-only |
| JDK | Temurin 21.0.13.1 | `winget install --id EclipseAdoptium.Temurin.21.JDK -e`. Your user-level `JAVA_HOME` points at Java 25, so set `JAVA_HOME` to JDK 21 in every shell that runs crforge (below) |
| crforge-env | 0.1.0, editable from `third_party/crforge/python` | JPype 1.7.1, NumPy 2.5.5 |

## Build crforge and its game tables

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.13.10002-hotspot"
$env:GRADLE_USER_HOME = "D:\ml-data\clash-ai\gradle-home"
cd third_party\crforge
.\gradlew.bat --no-daemon :env:installDist :tables:dataVersionTables "-Pcrforge.assetSource=cdn" "-Pcrforge.cache=D:/ml-data/clash-ai/crforge-cache"
```

- Builds natively on Windows in about 2.5 minutes; no WSL needed.
- `-Pcrforge.assetSource=cdn` downloads about 13 MB of the game's data files from Supercell's asset server, for client 16.635.13 / data version 16.635.8, and decodes them into 30 JSON tables. They stay on D: and are never committed.
- The asset server may stop serving this version after a game update, so the cache is backed up (zip SHA-256 `d4afae817da0439a628b74d9c5be66a717ef049dc8052112bc4686414f6a7714`). To restore, unzip it into `D:\ml-data\clash-ai\crforge-cache`.

## Python environment

```powershell
C:\Python312\python.exe -m venv D:\ml-data\clash-ai\venv
$py = "D:\ml-data\clash-ai\venv\Scripts\python.exe"
& $py -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install "third_party\crforge\python[dev]" pyarrow pyyaml
```

## Running crforge from Python

- crforge starts Java through JPype, which reads `JAVA_HOME`. Set it to JDK 21 first.
- Every entry point takes the installed engine folder, `third_party\crforge\env\build\install\env`, and the tables folder.
- On Windows, the JVM triggers access violations on purpose during start-up and handles them itself. Python's fault handler (on in pytest) prints them as "Windows fatal exception: access violation" with a stack dump. They're harmless; `-p no:faulthandler` silences them.
- pytest can't use the system temp folder on this machine: pass `-p no:cacheprovider --basetemp=D:/ml-data/clash-ai/tmp/pytest`.

## Verified (2026-11-24)

- `torch.cuda.is_available()` is true on the 3090 Ti.
- crforge's Python suite, with the real 16.635.8 tables: **283 passed**.

```powershell
cd third_party\crforge\python
& $py -m pytest -q -p no:cacheprovider -p no:faulthandler --basetemp=D:/ml-data/clash-ai/tmp/pytest-crforge "--install=..\env\build\install\env" "--tables=D:\ml-data\clash-ai\crforge-cache\tables\16.635.8"
```

- Throughput, `crforge-env-bench --engine ladder --decision-ticks 5` (random play, no neural network; one env step is one decision of one battle):

| threads | 64 battles | 256 battles |
|---:|---:|---:|
| 1 | 11,308 steps/s | 12,164 steps/s |
| 6 | 24,766 | 37,086 |
| 12 | 25,624 | 40,492 |

  At about 800 decisions a game, 40k steps/s is about 180,000 games an hour from the engine alone. Gains stop near 6 threads, which matches the 6 cores Windows sees.

## This project's package

```powershell
& $py -m pip install -e ".[dev]"
& $py -m pytest -q --basetemp=D:/ml-data/clash-ai/tmp/pytest
& $py -m clashai.eval.scorecard --policy scripted --opponent scripted --seeds 50
```

- `clashai` sets `JAVA_HOME` to JDK 21 itself (`clashai/paths.py`), so no shell setup is needed. Every location can be overridden with `CLASHAI_DATA`, `CLASHAI_CRFORGE_INSTALL`, `CLASHAI_TABLES` and `CLASHAI_JAVA_HOME`.
- Tests marked `engine` need the built engine and tables; without them they're skipped, not failed.
- Scorecards are written to `D:\ml-data\clash-ai\runs\scorecards`.

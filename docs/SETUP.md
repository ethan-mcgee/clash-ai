# Machine setup

How this project's Windows machine was set up for FirstLight_CR. FirstLight's own guide is [third_party/FirstLight_CR/docs/getting_started.md](../third_party/FirstLight_CR/docs/getting_started.md); this file records the choices and fixes specific to this setup.

## Hardware

- GPU: RTX 3090 Ti, 24 GB, NVIDIA driver 621.14
- CPU: Ryzen 9 7900X (Windows reports 12 logical processors)
- RAM: 64 GB
- Disks: C: is nearly full, so everything large goes on **D:**

## Layout

| What | Where | Why |
|---|---|---|
| This repo | OneDrive Desktop | Small; synced |
| FirstLight_CR | `third_party/FirstLight_CR`, git submodule pinned to `4c82496` | Apache-2.0 upstream, ~250 MB including shipped checkpoints |
| Python venv | `D:\ml-data\clash-ai\venv-firstlight` | Keeps multi-GB PyTorch out of OneDrive and off C: |
| Replay dataset | `D:\ml-data\clash-ai\datasets\royaleapi` | Same |
| FirstLight `.env` | `third_party/FirstLight_CR/.env` | Machine-specific; ignored by FirstLight's `.gitignore` |

## Tools

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12.6 (with Tkinter) | FirstLight targets 3.12 |
| PyTorch | 2.5.1, CUDA 12.4 build | Installed from `https://download.pytorch.org/whl/cu124`; the PyPI wheel on Windows is CPU-only |
| JDK | Temurin 17.0.21.1 | `winget install --id EclipseAdoptium.Temurin.17.JDK -e`. Another Java is first on PATH, so `.env` points at JDK 17 explicitly |
| Android SDK | `C:\Android\Sdk` | platform-tools 37.0.3, build-tools 37.0.0, NDK 27.3.13750724 (r27d) |
| MuMu Player | 6.8.15.0 | Instance 0, Android 12, root on, ARM64 via native bridge |

### Android SDK gotcha

`sdkmanager.bat` is deprecated, and Windows batch files split arguments on `;`, so `sdkmanager "build-tools;37.0.0"` silently installs nothing and still exits 0. Use the replacement CLI with `/`:

```powershell
$env:JAVA_HOME = "<JDK 17 folder>"
C:\Android\Sdk\cmdline-tools\latest\bin\android.exe --sdk=C:\Android\Sdk sdk install platform-tools build-tools/37.0.0 ndk/27.3.13750724
```

### Use one adb

MuMu ships its own `adb.exe`. Use the SDK's (`C:\Android\Sdk\platform-tools\adb.exe`) everywhere; two adb versions keep killing each other's server.

## `.env` keys filled

`CR_PYTHON`, `CR_PYTHONW`, `CR_REPLAY_DATASET_ROOT`, `CR_MUMU_MANAGER`, `CR_ADB`, `CR_VM_INDEX`, `CR_VM_NAME`, `CR_ADB_SERIAL`, `CR_JAVA`, `CR_KEYTOOL`, `CR_BUILD_TOOLS`, `CR_NDK_ROOT`. Everything else is left blank, as FirstLight's guide says.

## Game client

FirstLight needs the Null's Royale 15.535.13 arm64-v8a APK, supplied by the user at `third_party/FirstLight_CR/user-provided/nulls-royale.apk`. It is never committed: both repos' `.gitignore` files block `*.apk` and `user-provided/`.

## Verified

- `torch.cuda.is_available()` is true; a 4096×4096 matmul runs on the 3090 Ti.
- All five shipped checkpoints match the SHA-256 and sizes in `checkpoints/manifest.json`, and load with `weights_only=True` (plain tensors, no pickled code).
- FirstLight's `load_policy_v4` builds the General, IL and Hog 2.6 specialist 2 policies on CUDA: 12.05M parameters each, about 52 MB of VRAM for all three.

# crforge in this repo

[crforge](https://github.com/voonhous/crforge) (Apache-2.0) is vendored at `third_party/crforge` with `git subtree`: its files are ordinary files of this repo, so they can be read, changed and committed like our own.

| | |
|---|---|
| Upstream | `https://github.com/voonhous/crforge.git` |
| Vendored commit | `c56c196` (2026-11-22), squashed into one commit |
| Vendored on | 2026-11-24 |
| Line endings | left exactly as upstream (`.gitattributes`: `third_party/crforge/** -text`), so upstream pulls merge cleanly |

Build outputs (`build/`, `.gradle/`, the installed engine at `env/build/install/env`) and the game tables on D: are ignored by crforge's own `.gitignore` and never committed.

## Changing crforge

1. Edit the files under `third_party/crforge` and commit them with our other changes.
2. Mark every changed file with a line near its top, as Apache-2.0 section 4(b) requires, e.g. `// Modified for clash-ai: <what and why>`.
3. List the change in the table below.
4. Nothing to rebuild by hand for the engine or its Python package:
   - **Java** (`core`, `env` and the build files): `clashai` fingerprints these sources (their bytes, not file times) and, when they differ from the last build's, runs `gradlew :env:installDist` before starting the engine (`clashai.paths.ensure_engine`). Gradle compiles only what changed: about 30 s. `CLASHAI_NO_ENGINE_BUILD=1` turns this off.
   - **Python** (`python/crforge_env`): installed editable in the venv (`pip install --no-deps -e third_party\crforge\python`), so edits take effect on the next import.
   - **Tables** (`tables`): rebuild by hand with SETUP.md's build command.
5. Run crforge's tests for the part changed, then ours (`pytest`).

A change to the battle core breaks crforge's tick-exact match with the real game unless it fixes a mismatch. Keep the conformance tests passing.

## Our changes to crforge

| File | Change | Why | Commit |
|---|---|---|---|
| (none yet) | | | |

## Pulling upstream updates

```bash
git subtree pull --prefix=third_party/crforge https://github.com/voonhous/crforge.git main --squash
```

Then rebuild, rerun both test suites, and update the vendored commit above. Our changes merge like any other branch's, and a conflict means upstream changed a line we changed. A newer crforge may target a newer game data version: check `crforge-data.lock` before pulling, since this project's tables and replays are 16.635.8.

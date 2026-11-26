"""What a training run that lasts days needs: no sleep, no lost checkpoints, a log on disk.

- :func:`keep_awake` stops Windows sleeping while the process lives
- :func:`atomic_save` writes a checkpoint to a temporary file and renames it over the old one, so
  a run killed mid-save still has its last complete checkpoint
- :func:`tee` copies everything printed to a log file as well, so a run started hidden (no
  console window to close by mistake) still has its output on disk
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import torch

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake() -> None:
    """No idle sleep while this process lives (Windows; elsewhere nothing)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def atomic_save(obj, path: Path) -> None:
    path = Path(path)
    part = path.with_name(path.name + ".part")
    torch.save(obj, part)
    os.replace(part, path)


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        if self.stream is not None:
            try:
                self.stream.write(text)
            except (OSError, ValueError):
                self.stream = None  # the console went away: keep logging to the file
        self.log.write(text)
        return len(text)

    def flush(self):
        if self.stream is not None:
            try:
                self.stream.flush()
            except (OSError, ValueError):
                self.stream = None
        self.log.flush()


def tee(path: Path) -> None:
    """From now on, everything printed (and errors) also goes to ``path``, appended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 (open for the process)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)

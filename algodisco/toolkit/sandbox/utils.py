# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import multiprocessing
import os
import sys
import functools

from typing import Literal


def _set_mp_start_method(
    multiprocessing_start_method: Literal["default", "auto", "fork", "spawn"],
):
    if multiprocessing_start_method == "auto":
        # Force macOS and Linux use 'fork' to generate new process
        if sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
            multiprocessing.set_start_method("fork", force=True)
    elif multiprocessing_start_method == "fork":
        multiprocessing.set_start_method("fork", force=True)
    elif multiprocessing_start_method == "spawn":
        multiprocessing.set_start_method("spawn", force=True)


def _redirect_to_devnull():
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

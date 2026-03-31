# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any
from algodisco.base.logger import AlgoSearchLoggerBase


class BasePickleLogger(AlgoSearchLoggerBase):
    """A thread-safe logger that writes data directly to pickle files per item category."""

    def __init__(self, logdir: str):
        """
        Initializes the pickle logger with a directory.

        Args:
            logdir: The directory where pickle files will be stored.
        """
        self._logdir = logdir
        self._lock = Lock()
        # External backends such as W&B/SwanLab expect one monotonic step stream
        # for the whole run, even when multiple item types are interleaved.
        self._global_step = 0
        self._counters: Dict[str, int] = {}  # item count within each batch
        self._batch_counters: Dict[str, int] = {}  # batch number for each item_name
        self._caches: Dict[str, list] = {}
        self._flush_frequencies: Dict[str, int] = (
            {}
        )  # flush frequency for each item_name
        self._flush_counts: Dict[str, int] = (
            {}
        )  # count since last flush for each item_name
        if os.path.exists(self._logdir):
            import shutil

            shutil.rmtree(self._logdir)
        os.makedirs(self._logdir, exist_ok=True)

    def set_log_item_flush_frequency(self, frequencies: Dict[str, int]):
        """
        Set the flush frequency for each item_name.

        Args:
            frequencies: Dict mapping item_name to flush frequency.
                e.g., {"algo": 2000, "database": 1} means:
                - flush "algo" every 2000 items
                - flush "database" every 1 item (i.e., always flush immediately)
        """
        with self._lock:
            self._flush_frequencies.update(frequencies)
            # Initialize flush counts for new items
            for item_name in frequencies:
                if item_name not in self._flush_counts:
                    self._flush_counts[item_name] = 0

    def _get_item_dir(self, item_name: str) -> Path:
        """Get the directory for a specific item_name."""
        return Path(self._logdir) / item_name

    def _get_item_path(self, item_name: str, batch_num: int) -> Path:
        """Get the path for a specific batch file."""
        return self._get_item_dir(item_name) / f"{batch_num}.pkl"

    def _pre_log_hook(self, log_item: Dict, item_name: str, *, count: int, step: int):
        """A hook executed before an item is logged. Assumes lock is held."""
        pass

    def _flush(self, item_name: str):
        """Flushes the cache for item_name to disk. Assumes lock is held."""
        if not self._caches.get(item_name) or not self._caches[item_name]:
            return

        # Ensure directory exists
        item_dir = self._get_item_dir(item_name)
        item_dir.mkdir(parents=True, exist_ok=True)

        # Get batch number and increment for next flush
        batch_num = self._batch_counters.get(item_name, 1)
        self._batch_counters[item_name] = batch_num + 1

        # Save all items in cache as a single pickle file
        path = self._get_item_path(item_name, batch_num)
        with open(path, "wb") as f:
            pickle.dump(self._caches[item_name], f)

        # Clear cache and reset flush count
        self._caches[item_name] = []
        self._flush_counts[item_name] = 0

    async def log_dict(self, log_item: Dict, item_name: str):
        """Logs a dictionary to the cache and flushes based on frequency."""
        self.log_dict(log_item, item_name)

    async def finish(self):
        """Flush all remaining items in the caches."""
        self.finish()

    def log_dict(self, log_item: Dict, item_name: str):
        """Synchronous version of log_dict."""
        with self._lock:
            if item_name not in self._counters:
                self._counters[item_name] = 1
                self._batch_counters[item_name] = 1
                self._caches[item_name] = []
                self._flush_counts[item_name] = 0

            count = self._counters[item_name]
            step = self._global_step + 1

            log_item["count"] = count
            self._pre_log_hook(log_item, item_name, count=count, step=step)

            self._counters[item_name] = count + 1
            self._global_step = step

            self._caches[item_name].append(log_item)

            # Increment flush count and check if we should flush
            self._flush_counts[item_name] = self._flush_counts.get(item_name, 0) + 1
            # Default frequency is 1 (flush every time) if not set
            frequency = self._flush_frequencies.get(item_name, 1)

            if self._flush_counts[item_name] >= frequency:
                self._flush(item_name)

    def finish(self):
        """Synchronous version of finish."""
        with self._lock:
            for item_name in list(self._caches.keys()):
                self._flush(item_name)

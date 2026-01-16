import os
import pickle
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Optional


class PickleLogger:
    """A thread-safe logger that caches data and writes to a pickle file periodically."""

    def __init__(
        self,
        logdir: PathLike | str,
        file_name: str = "search_log.pkl",
        write_file_frequency: Optional[int] = 100,
    ):
        """Initializes the logger.

        Args:
            logdir: The directory to save the log file.
            file_name: The name of the log file. Defaults to 'search_log.pkl'.
            write_file_frequency: How often to write the cache to the file. If None, only writes on `finish()`.
        """
        self._counter = 1
        self._logdir = Path(logdir)
        self._file_name = (
            file_name if file_name.endswith(".pkl") else file_name + ".pkl"
        )
        self._write_file_frequency = write_file_frequency
        self._lock = Lock()
        self._cache = []

        os.makedirs(self._logdir, exist_ok=True)
        # Initialize with an empty list
        with open(self._logdir / self._file_name, "wb") as f:
            pickle.dump([], f)

    def _pre_log_hook(self, log_dict: dict):
        """A hook executed before an item is logged to the cache. Assumes lock is held."""
        pass

    def log_to_cache(self, log_dict: dict):
        """Logs a dictionary to the cache and periodically writes to the file.

        This method is thread-safe.
        """
        with self._lock:
            self._pre_log_hook(log_dict)

            log_dict["count"] = self._counter
            self._cache.append(log_dict)
            if (
                self._write_file_frequency
                and self._counter % self._write_file_frequency == 0
            ):
                self._write_cache_to_file_nolock()
            self._counter += 1

    def _write_cache_to_file_nolock(self):
        """Writes the cache to the file. Assumes the lock is already held."""
        if not self._cache:
            return

        local_cache = self._cache
        self._cache = []

        try:
            with open(self._logdir / self._file_name, "rb") as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError):
            data = []

        data.extend(local_cache)

        with open(self._logdir / self._file_name, "wb") as f:
            pickle.dump(data, f)

    def write_cache_to_file(self):
        """Explicitly writes the current cache to the file."""
        with self._lock:
            self._write_cache_to_file_nolock()

    def finish(self):
        """Writes any remaining items in the cache to the file."""
        self.write_cache_to_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass


class PickleLoggerWithSwanLab(PickleLogger):
    """A Pickle logger that also logs metrics to SwanLab."""

    def __init__(
        self,
        logdir: PathLike | str,
        project: str,
        file_name: str = "search_log.pkl",
        experiment_name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[dict] = None,
        swanlab_logdir: Optional[PathLike | str] = None,
        api_key: Optional[str] = None,
        write_file_frequency: Optional[int] = 100,
    ):
        """Initializes the logger and a SwanLab run.

        Args:
            logdir: The directory to save the local pickle log file.
            project: The name of the SwanLab project.
            file_name: The name of the log file. Defaults to 'search_log.pkl'.
            experiment_name: The name for the SwanLab run. Defaults to None, letting SwanLab decide.
            group: The group for the SwanLab run. Defaults to None, letting SwanLab decide.
            config: A dictionary of hyperparameters for the run. Defaults to None.
            swanlab_logdir: The directory for SwanLab's internal logs. If None, SwanLab uses its default ("swanlog").
            api_key: The SwanLab API key. Defaults to None.
            write_file_frequency: How often to write the cache to the local file.
        """
        super().__init__(
            logdir, file_name=file_name, write_file_frequency=write_file_frequency
        )

        import swanlab

        if api_key:
            swanlab.login(api_key)

        swanlab_init_kwargs = {
            "project": project,
            "experiment_name": experiment_name,
            "group": group,
            "config": config,
        }

        if swanlab_logdir is not None:
            swanlab_init_kwargs["logdir"] = str(swanlab_logdir)

        # Filter out None values so swanlab can use its defaults for experiment_name, group, config
        swanlab_init_kwargs = {
            k: v for k, v in swanlab_init_kwargs.items() if v is not None
        }

        self._logger = swanlab.init(**swanlab_init_kwargs)
        if not self._logger:
            # swanlab.init returns None if it's disabled. Create a dummy logger.
            class DummyLogger:
                def log(self, *args, **kwargs):
                    pass

                def finish(self, *args, **kwargs):
                    pass

            self._logger = DummyLogger()

        self._best_score = -float("inf")
        self._all_scores = []
        self._cumulative_sample_time = 0.0
        self._cumulative_eval_time = 0.0
        self._database_ref = None
        self._valid_functions_num = 0
        self._invalid_functions_num = 0

    def set_database(self, db):
        """Sets a reference to the AlgoDatabase to query for stats."""
        self._database_ref = db

    def _pre_log_hook(self, log_dict: dict):
        """Logs metrics to swanlab before caching."""
        log_items = {}

        # 1. Update state with the current sample's data
        score = log_dict.get("score")
        if score is not None:
            self._valid_functions_num += 1
            self._all_scores.append(score)
            if score > self._best_score:
                self._best_score = score
        else:
            self._invalid_functions_num += 1

        if "sample_time" in log_dict:
            self._cumulative_sample_time += log_dict["sample_time"]
        if "eval_time" in log_dict:
            self._cumulative_eval_time += log_dict["eval_time"]

        # 2. Prepare items for SwanLab logging
        # Best score so far
        if self._best_score > -float("inf"):
            log_items["best_score"] = self._best_score

        # Top-K average scores
        self._all_scores.sort(reverse=True)
        for k in [5, 10, 20, 30]:
            if len(self._all_scores) >= k:
                top_k_avg = sum(self._all_scores[:k]) / k
                log_items[f"top_{k}_avg_score"] = top_k_avg

        # Cumulative times
        log_items["cumulative_sample_time"] = self._cumulative_sample_time
        log_items["cumulative_eval_time"] = self._cumulative_eval_time

        # Island stats
        if self._database_ref:
            island_stats = self._database_ref.get_island_stats()
            if island_stats:
                for island_id, size in island_stats.items():
                    log_items[f"island_size_{island_id}"] = size

        # Log other numeric values from the original log_dict, excluding island_id
        for k, v in log_dict.items():
            if isinstance(v, (int, float)) and k != "island_id":
                log_items[k] = v

        log_items["num_valid_functions"] = self._valid_functions_num
        log_items["num_invalid_functions"] = self._invalid_functions_num

        self._logger.log(log_items, step=self._counter)

    def finish(self):
        super().finish()
        if hasattr(self._logger, "finish"):
            self._logger.finish()

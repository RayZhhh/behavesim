# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import logging
from os import PathLike
from typing import Optional, Dict
from .pickle_logger import BasePickleLogger


class BaseSwanLabLogger(BasePickleLogger):
    """A base Pickle logger that also logs common metrics to SwanLab."""

    def _init_swanlab(self):
        import swanlab

        if self._initialized:
            return

        existing_run = swanlab.get_run()
        if existing_run is not None:
            logging.warning(
                "SwanLab has already been initialized; BaseSwanLabLogger will reuse the existing run."
            )
            self._swanlab_run = existing_run
            self._owns_swanlab_run = False
            self._initialized = True
            return

        if self._api_key:
            swanlab.login(self._api_key)

        swanlab_init_kwargs = {
            "project": self._project_name,
            "experiment_name": self._experiment_name,
            "group": self._group,
            "config": self._config,
        }

        # Use the explicit SwanLab logdir when provided, otherwise fall back
        # to the pickle logger directory.
        swanlab_logdir = (
            self._swanlab_logdir
            if self._swanlab_logdir is not None
            else self._logdir
        )
        if swanlab_logdir is not None:
            swanlab_init_kwargs["logdir"] = str(swanlab_logdir)

        swanlab_init_kwargs = {
            k: v for k, v in swanlab_init_kwargs.items() if v is not None
        }

        self._swanlab_run = swanlab.init(**swanlab_init_kwargs)
        if not self._swanlab_run:

            class DummyLogger:
                def log(self, *args, **kwargs):
                    pass

                def finish(self, *args, **kwargs):
                    pass

            self._swanlab_run = DummyLogger()
            self._owns_swanlab_run = False
        else:
            self._owns_swanlab_run = True

        self._initialized = True

    def __init__(
        self,
        logdir: PathLike | str,
        project: str,
        experiment_name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[dict] = None,
        swanlab_logdir: Optional[PathLike | str] = None,
        api_key: Optional[str] = None,
        *,
        lazy_init: bool = False,
    ):
        super().__init__(
            logdir=str(logdir),
        )

        self._project_name = project
        self._experiment_name = experiment_name
        self._group = group
        self._config = config
        self._swanlab_logdir = swanlab_logdir
        self._api_key = api_key
        self._swanlab_run = None
        self._owns_swanlab_run = False
        self._initialized = False
        self._lazy_init = lazy_init

        try:
            import swanlab  # noqa: F401
        except ImportError:
            raise ImportError(
                "SwanLab is not installed. Please install it with 'pip install swanlab'"
            )

        if not self._lazy_init:
            self._init_swanlab()

        self._best_score = -float("inf")
        self._all_scores = []
        self._cumulative_sample_time = 0.0
        self._cumulative_eval_time = 0.0
        self._cumulative_execution_time = 0.0
        self._valid_functions_num = 0
        self._invalid_functions_num = 0

    def _prepare_swanlab_log_items(
        self, log_dict: dict, item_name: str
    ) -> Optional[dict]:
        """Prepare SwanLab metrics for per-algorithm search logs."""
        # Database snapshots are still written to pickle by the base logger,
        # but they should not share the same SwanLab metric names/steps.
        if item_name != "algo":
            return None

        if not self._initialized:
            self._init_swanlab()

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
        if "execution_time" in log_dict:
            self._cumulative_execution_time += log_dict["execution_time"]

        # 2. Prepare items for SwanLab logging
        if self._best_score > -float("inf"):
            log_items["best_score"] = self._best_score

        self._all_scores.sort(reverse=True)
        for k in [5, 10, 20, 30]:
            if len(self._all_scores) >= k:
                top_k_avg = sum(self._all_scores[:k]) / k
                log_items[f"top_{k}_avg_score"] = top_k_avg

        log_items["cumulative_sample_time"] = self._cumulative_sample_time
        log_items["cumulative_eval_time"] = self._cumulative_eval_time
        log_items["cumulative_execution_time"] = self._cumulative_execution_time

        # Log other numeric values from the original log_dict
        for k, v in log_dict.items():
            if isinstance(v, (int, float)):
                log_items[k] = v

        log_items["num_valid_functions"] = self._valid_functions_num
        log_items["num_invalid_functions"] = self._invalid_functions_num

        return log_items

    def _pre_log_hook(self, log_item: Dict, item_name: str, *, count: int, step: int):
        """Logs metrics to swanlab before caching."""
        log_items = self._prepare_swanlab_log_items(log_item, item_name)
        if not log_items:
            return

        sample_num = log_item.get("sample_num")
        if isinstance(sample_num, (int, float)):
            step = int(sample_num)

        self._swanlab_run.log(log_items, step=step)

    async def finish(self):
        await super().finish()
        if self._owns_swanlab_run and hasattr(self._swanlab_run, "finish"):
            self._swanlab_run.finish()

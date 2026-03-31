# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import threading
import numpy as np
from typing import List, Optional, Any, Dict
from algodisco.base.algo import AlgoProto


class EoHDatabase:
    """Population management for EoH."""

    def __init__(self, pop_size: int):
        self._pop_size = pop_size
        self._population: List[AlgoProto] = []
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._population)

    def register_algo(self, algo: AlgoProto):
        """Adds a new algorithm to the population and performs survival if full."""
        with self._lock:
            if algo.score is None:
                return

            # Avoid duplicates by code or score
            if any(
                str(p.program) == str(algo.program) or p.score == algo.score
                for p in self._population
            ):
                return

            self._population.append(algo)
            self._survival()

    def _survival(self):
        """Keeps only the top pop_size individuals."""
        if len(self._population) > self._pop_size:
            # Sort by score descending
            self._population.sort(key=lambda x: x.score, reverse=True)
            self._population = self._population[: self._pop_size]

    def select_algos(self, k: int) -> List[AlgoProto]:
        """Rank-based selection as per EoH paper."""
        with self._lock:
            if not self._population:
                return []

            # Filter out inf scores
            valid_algos = [a for a in self._population if not np.isinf(a.score)]
            if not valid_algos:
                return []

            # Sort by score descending
            sorted_algos = sorted(valid_algos, key=lambda x: x.score, reverse=True)

            n = len(sorted_algos)
            # Probability proportional to 1 / (rank + n)
            p = [1.0 / (i + 1 + n) for i in range(n)]
            p = np.array(p)
            p /= p.sum()

            indices = np.random.choice(n, size=min(k, n), p=p, replace=False)
            return [sorted_algos[i] for i in indices]

    def get_best_score(self) -> float:
        with self._lock:
            if not self._population:
                return -float("inf")
            return max(a.score for a in self._population)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"population": [a.to_dict() for a in self._population]}

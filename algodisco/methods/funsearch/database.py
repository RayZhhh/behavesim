# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import copy
import time
import threading
from typing import Any, Tuple, List, Optional

import numpy as np
import scipy

from algodisco.base.algo import AlgoProto


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite 'logits'."""
    if not np.all(np.isfinite(logits)):
        non_finite = set(logits[~np.isfinite(logits)])
        raise ValueError(f'"logits" contains non-finite value(s): {non_finite}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        num_islands: int = 10,
        max_island_capacity: Optional[int] = None,
        reset_period: int = 4 * 60 * 60,
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30_000,
    ) -> None:
        self._num_islands = num_islands
        self._max_island_capacity = max_island_capacity
        self._reset_period = reset_period
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._islands: list[Island] = [
            Island(
                max_capacity=max_island_capacity,
                cluster_sampling_temperature_init=cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=cluster_sampling_temperature_period,
            )
            for _ in range(num_islands)
        ]
        self._best_score_per_island: list[float] = [-float("inf")] * num_islands
        self._best_program_per_island: list[AlgoProto | None] = [None] * num_islands
        self._last_reset_time: float = time.time()
        self._lock = threading.RLock()

    def select_programs(self, num_programs: int) -> Tuple[List[AlgoProto], int]:
        """Select programs from database, return them with their island id."""
        with self._lock:
            island_id = np.random.randint(len(self._islands))
            programs = self._islands[island_id].select_programs(num_programs)
            return programs, island_id

    def gather_all_programs_grouped_by_islands(self) -> List[List[AlgoProto]]:
        with self._lock:
            all_programs = []
            for island in self._islands:
                all_programs.append(island.programs)
            return all_programs

    def get_island_stats(self) -> dict[int, int]:
        """Returns a dictionary with the number of programs in each island."""
        with self._lock:
            return {i: len(island) for i, island in enumerate(self._islands)}

    def get_all_algorithms(self) -> List[AlgoProto]:
        """Gathers all algorithms from all islands in the database."""
        all_algos: List[AlgoProto] = []
        with self._lock:
            for island in self._islands:
                all_algos.extend(island.programs)
        return all_algos

    def to_dict(self) -> dict[str, Any]:
        """Converts the ProgramsDatabase to a dictionary."""
        with self._lock:
            return {"islands": [island.to_dict() for island in self._islands]}

    def _register_program_in_island(
        self,
        program: AlgoProto,
        island_id: int,
    ) -> None:
        """Registers 'program' to the specified island."""
        score = program.score
        self._islands[island_id].register_program(program)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_score_per_island[island_id] = score

    def register_program(
        self,
        program: AlgoProto,
        island_id: int | None,
    ):
        """Registers 'program' in the database."""
        with self._lock:
            program = copy.deepcopy(program)
            if island_id is None:
                # This is a program added at the beginning, so adding it to all islands
                for island_id in range(len(self._islands)):
                    self._register_program_in_island(program, island_id)
            else:
                self._register_program_in_island(program, island_id)

            # Check whether it is time to reset an island
            if time.time() - self._last_reset_time > self._reset_period:
                self._last_reset_time = time.time()
                self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island
            + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                max_capacity=self._max_island_capacity,
                cluster_sampling_temperature_init=self._cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=self._cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float("inf")
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            if founder:
                self._register_program_in_island(founder, island_id)


class Island:
    """A sub-population of the program database."""

    def __init__(
        self,
        max_capacity: Optional[int] = None,
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30_000,
    ) -> None:
        self._max_capacity = max_capacity
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._clusters: dict[Any, Cluster] = {}
        self._num_programs: int = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return self._num_programs

    @property
    def programs(self) -> List[AlgoProto]:
        with self._lock:
            all_programs = []
            for cluster in self._clusters.values():
                all_programs.extend(cluster.programs)
            return all_programs

    def to_dict(self) -> dict[str, Any]:
        """Converts the Island to a dictionary."""
        with self._lock:
            return {
                "clusters": [cluster.to_dict() for cluster in self._clusters.values()]
            }

    def register_program(self, program: AlgoProto):
        """Stores a program on this island, in its appropriate cluster."""
        with self._lock:
            signature = program.score
            if signature not in self._clusters:
                self._clusters[signature] = Cluster(program.score, program)
            else:
                self._clusters[signature].register_program(program)
            self._num_programs += 1

            # Check if capacity exceeded and perform survival if needed
            if (
                self._max_capacity is not None
                and self._num_programs > self._max_capacity
            ):
                self._survival()

    def _survival(self):
        """Performs survival: keeps the highest-scoring programs up to max_capacity."""
        if self._max_capacity is None:
            return

        # Get all programs sorted by score (descending)
        all_programs = []
        for cluster in self._clusters.values():
            all_programs.extend(cluster.programs)

        # Sort by score descending
        all_programs.sort(
            key=lambda p: p.score if p.score is not None else float("-inf"),
            reverse=True,
        )

        # Keep only top max_capacity programs
        kept_programs = all_programs[: self._max_capacity]

        # Rebuild clusters with only kept programs
        self._clusters = {}
        for program in kept_programs:
            signature = program.score
            if signature not in self._clusters:
                self._clusters[signature] = Cluster(program.score, program)
            else:
                self._clusters[signature].register_program(program)

        self._num_programs = len(kept_programs)

    def select_programs(self, num_programs: int) -> list[AlgoProto]:
        """Select programs on this island, return programs sorted by their fitness score."""
        with self._lock:
            if not self._clusters:
                return []

            signatures = list(self._clusters.keys())
            cluster_scores = np.array(
                [self._clusters[signature].score for signature in signatures]
            )
            # Normalized the score
            max_abs_score = float(np.abs(cluster_scores).max())
            if max_abs_score > 1:
                cluster_scores = cluster_scores.astype(float) / max_abs_score

            # Convert scores to probabilities using softmax with temperature schedule
            period = self._cluster_sampling_temperature_period
            temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period
            )
            probabilities = _softmax(cluster_scores, temperature)

            # At the beginning of an experiment when we have few clusters,
            # place fewer programs into the prompt
            programs_per_prompt = min(len(self._clusters), num_programs)

            idx = np.random.choice(
                len(signatures),
                size=programs_per_prompt,
                p=probabilities,
                replace=False,
            )
            chosen_signatures = [signatures[i] for i in idx]
            implementations = []
            scores = []
            for signature in chosen_signatures:
                cluster = self._clusters[signature]
                implementations.append(cluster.sample_program())
                scores.append(cluster.score)

            indices = np.argsort(scores)
            sorted_implementations = [implementations[i] for i in indices]
            return sorted_implementations


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: AlgoProto):
        self._score = score
        self._programs: list[AlgoProto] = [implementation]
        self._lengths: list[int] = [len(str(implementation.program))]
        self._lock = threading.RLock()

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    @property
    def programs(self) -> List[AlgoProto]:
        with self._lock:
            return self._programs

    def to_dict(self) -> dict[str, Any]:
        """Converts the Cluster to a dictionary."""
        with self._lock:
            return {
                "score": self._score,
                "programs": [p.to_dict() for p in self._programs],
            }

    def register_program(self, program: AlgoProto) -> None:
        """Adds 'program' to the cluster."""
        with self._lock:
            self._programs.append(program)
            self._lengths.append(len(str(program.program)))

    def sample_program(self) -> AlgoProto:
        """Samples a program, giving higher probability to shorter programs."""
        with self._lock:
            normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6
            )
            if len(normalized_lengths) == 1:
                return self._programs[0]
            probabilities = _softmax(-normalized_lengths, temperature=1.0)
            return np.random.choice(self._programs, p=probabilities)


if __name__ == "__main__":
    import random

    def rand_algo_str():
        algo = "def f():\n"
        l = random.randint(10, 120)
        for i in range(l):
            algo += "    return x + 1\n"
        return algo

    def get_rand_algo():
        algo = AlgoProto.from_program_str(rand_algo_str())
        algo.score = random.randint(-9000, 0)
        return algo

    db = ProgramsDatabase()
    for _ in range(100000):
        algo = get_rand_algo()
        db.register_program(algo, island_id=random.randint(0, 9))
        db.select_programs(num_programs=2)
        print(len(db.get_all_algorithms()))

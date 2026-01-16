import dataclasses
import functools
import logging
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from behavesim_search.algo_proto import AlgoProto
from behavesim_search.similarity_calculator import BehaveSimCalculator


def _rank_based_selection(
    scores: List[float], choices: List[Any], k: int, exploitation_intensity: float = 1.0
) -> List[Any]:
    r"""Selects k items based on rank, with higher scores being more likely.

    The selection probability for each item $i$ is given by
    $p_i = \frac{r_i^{-\alpha}}{\sum_{j=1}^n r_j^{-\alpha}}$,
    where $r_i$ is the rank of item $i$ (1 for the highest score) and $\alpha$
    (exploitation_intensity) controls the selection pressure.

    Args:
        scores: A list of scores corresponding to the choices.
        choices: A list of items to choose from.
        k: The number of items to select.
        exploitation_intensity: A non-negative float controlling the selection
            pressure.
            - If 0, sampling is uniform (all items have equal probability).
            - As it approaches infinity, it implements hill-climbing
              (only the top-ranked item is selected).

    Returns:
        A list of k selected items.
    """
    if len(scores) != len(choices):
        raise ValueError("Length of scores and choices must be the same.")
    if k > len(choices):
        raise ValueError("k cannot be greater than the number of choices.")
    if exploitation_intensity < 0:
        raise ValueError("exploitation_intensity cannot be negative.")

    # Sort choices based on scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_choices = [choices[i] for i in sorted_indices]

    # Assign ranks (1-based)
    ranks = np.arange(1, len(sorted_choices) + 1)

    # Calculate probabilities based on the provided formula
    if exploitation_intensity == 0:
        probabilities = np.ones_like(ranks, dtype=float)  # Uniform sampling
    else:
        probabilities = ranks ** (-exploitation_intensity)
    probabilities /= np.sum(probabilities)

    # Select k items based on rank probabilities
    selected_indices = np.random.choice(
        len(sorted_choices), size=k, p=probabilities, replace=False
    )
    selected_items = [sorted_choices[i] for i in selected_indices]

    return selected_items


def _compute_avg_sim_for_island(
    algo: AlgoProto,
    island_algos: List[AlgoProto],
    sim_calculator: BehaveSimCalculator,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Calculates the average similarity for a candidate algorithm against an island."""

    if not island_algos:
        return 0.0, {}, {}

    similarities = []

    # The sim cache records {k: "target algo_proto_id", v: "similarity between algo and target algo"}
    # Sim cache is used for calculating the dissimilarity value for two algos within same islands,
    # avoiding redundant re-calculation
    sim_cache = {}

    # Timings are recorded for debug
    # Sometimes computing similarity is time-consuming while registering algorithms
    total_timings = {}

    for island_algo in island_algos:
        try:
            sim, timings = sim_calculator.calculate_sim(
                algo,
                island_algo,
            )
            # Skip unexpected value encountered
            # Set the similarity to 0.0
            if np.isnan(sim) or np.isinf(sim):
                sim = 0.0
            # Aggregate timings
            for k, v in timings.items():
                total_timings[k] = total_timings.get(k, 0.0) + v
        except Exception:
            # Set the similarity to 0.0 if getting an exception
            sim = 0.0

        similarities.append(sim)
        sim_cache[island_algo.id] = sim

    return float(np.mean(similarities)), sim_cache, total_timings


def _get_algo_sim(
    algo1: AlgoProto, algo2: AlgoProto, sim_calculator: BehaveSimCalculator
) -> float:
    # We first try to get similarity from sim cache to avoid re-calculation
    sim_cache2 = algo2["sim_cache"]
    sim_cache1 = algo1["sim_cache"]

    if algo1.id in sim_cache2:
        sim = sim_cache2[algo1.id]
    elif algo2.id in sim_cache1:
        sim = sim_cache1[algo2.id]
    else:
        # If similarity cannot be found, then calculate
        try:
            sim, _ = sim_calculator.calculate_sim(
                algo1,
                algo2,
            )
            # Assign the similarity to zero for strange value
            if sim is None or np.isnan(sim) or np.isinf(sim):
                sim = 0.0
        except:  # noqa
            # Assign the similarity to zero when encountering exception
            sim = 0.0
        # Save to sim cache
        algo1["sim_cache"][algo2.id] = sim
        algo2["sim_cache"][algo1.id] = sim

    return sim


@dataclasses.dataclass
class AlgoDatabaseConfig:
    algo_sim_calculator: BehaveSimCalculator = BehaveSimCalculator()

    # --- Island specifications ---
    n_islands: int = 10
    island_capacity: Optional[int] = 80  # None means an endless large island

    # --- Selection parameters ---
    selection_exploitation_intensity: float = 0.9

    # --- Algo registration acceleration parameters ---
    num_sim_caculator_workers: int = 4
    async_register: bool = True


class AlgoDatabase:
    def __init__(
        self,
        algo_database_config: AlgoDatabaseConfig = AlgoDatabaseConfig(),
        islands: List["AlgoIsland"] = None,
    ):
        """Initializes the AlgoDatabase.

        Args:
            algo_database_config: Configuration for the algorithm database.
            islands: An optional list of pre-initialized AlgoIsland instances.
                     If None, `n_islands` new AlgoIsland instances are created
                     based on `algo_database_config`.
        """
        self.algo_database_config = algo_database_config

        if islands is not None:
            self.islands: List[AlgoIsland] = islands
        else:
            self.islands: List[AlgoIsland] = [
                AlgoIsland(
                    sim_calculator=algo_database_config.algo_sim_calculator,
                    island_capacity=algo_database_config.island_capacity,
                    exploitation_intensity=algo_database_config.selection_exploitation_intensity,
                )
                for _ in range(algo_database_config.n_islands)
            ]
        self._lock = threading.RLock()
        self._last_reset_time = time.time()

        # Executor for coordinating the registration process (running in a background thread)
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Executor for CPU-bound similarity calculations (running in background processes)
        self._process_executor = ProcessPoolExecutor(
            max_workers=self.algo_database_config.num_sim_caculator_workers
        )

    def register_algo(self, algo: AlgoProto):
        """Registers the given algorithm to the most similar island asynchronously.

        This method submits the registration task to a background thread pool, allowing
        the caller to proceed immediately. The background task handles similarity
        calculation (utilizing multiple processes) and then acquires the
        lock to place the algorithm into the chosen island.

        Args:
            algo: The algorithm prototype to register.
        """
        future = self._executor.submit(self._register_algo_worker, algo)
        if not self.algo_database_config.async_register:
            future.result()

    def cluster_and_reassign_islands(self):
        """Clusters all algorithms and reassigns them to islands."""
        with self._lock:
            n_islands = self.algo_database_config.n_islands
            # 1. Get all algorithms from all islands
            all_algos = [
                algo for island in self.islands for algo in island.get_all_algorithms()
            ]

            if len(all_algos) < n_islands:
                logging.warning(
                    f"Not enough algorithms ({len(all_algos)}) to form {n_islands} clusters. Skipping reassignment."
                )
                return

            # 2. Calculate similarity matrix
            num_algos = len(all_algos)
            sim_matrix = np.zeros((num_algos, num_algos))
            for i in range(num_algos):
                for j in range(i, num_algos):
                    if i == j:
                        sim = 1.0
                    else:
                        sim = _get_algo_sim(
                            all_algos[i],
                            all_algos[j],
                            self.algo_database_config.algo_sim_calculator,
                        )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            # 3. Perform clustering
            # Convert similarity to distance
            dist_matrix = 1 - sim_matrix

            from sklearn.cluster import KMeans

            # Perform clustering using KMeans as per the user's example
            kmeans = KMeans(n_clusters=n_islands, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(dist_matrix)

            # 4. Reassign algorithms to islands
            # Create new empty islands
            new_islands = [
                AlgoIsland(
                    sim_calculator=self.algo_database_config.algo_sim_calculator,
                    island_capacity=self.algo_database_config.island_capacity,
                    exploitation_intensity=self.algo_database_config.selection_exploitation_intensity,
                )
                for _ in range(n_islands)
            ]

            # Assign algorithms to new islands based on cluster labels
            for algo, label in zip(all_algos, cluster_labels):
                # sklearn cluster labels are 0-based
                if 0 <= label < n_islands:
                    new_islands[label].register_algo(algo)

            self.islands = new_islands
            logging.info(
                f"Successfully clustered {num_algos} algorithms into {n_islands} islands."
            )

    def _register_algo_worker(self, algo: AlgoProto):
        """Worker method for asynchronous algorithm registration."""
        if algo.score is None:
            return

        # Snapshot islands for calculation to avoid holding lock during heavy calc.
        with self._lock:
            if not self.islands:
                return

            # Prefer empty islands to ensure diversity and exploration.
            empty_island_indices = [
                i for i, island in enumerate(self.islands) if not island
            ]
            if empty_island_indices:
                chosen_island_index = random.choice(empty_island_indices)
                self.islands[chosen_island_index].register_algo(algo)
                self.restart_database()
                return

            # Take a snapshot of islands for similarity calculation
            islands_snapshot = list(self.islands)

        # Calculate similarity against each island in parallel using processes
        futures = []
        for island in islands_snapshot:
            # Pass only data needed for calculation to picklable worker
            futures.append(
                self._process_executor.submit(
                    _compute_avg_sim_for_island,
                    algo,
                    island.algorithms,
                    self.algo_database_config.algo_sim_calculator,
                )
            )

        island_results = [f.result() for f in futures]
        island_similarities = [res[0] for res in island_results]

        # Save similarity results in the "sim_cache"
        sim_cache_all_island = [res[1] for res in island_results]
        sim_caches_all_island = functools.reduce(dict.__or__, sim_cache_all_island, {})  # noqa
        algo["sim_cache"] = sim_caches_all_island

        # Aggregate timings across all islands
        all_timings = {}
        for _, _, timings in island_results:
            for k, v in timings.items():
                all_timings[k] = all_timings.get(k, 0.0) + v

        if all_timings:
            timing_str = ", ".join([f"{k}: {v:.4f}s" for k, v in all_timings.items()])
            logging.info(
                f">>> [{self.__class__.__name__}] "
                f"Total Similarity Calculation Times (Register): {timing_str}",
            )

        # Handle potential NaNs in similarities
        clean_similarities = [
            (sim if not np.isnan(sim) else -float("inf")) for sim in island_similarities
        ]

        max_sim = max(clean_similarities)
        candidate_indices = [
            i for i, sim in enumerate(clean_similarities) if sim == max_sim
        ]

        chosen_island_index = random.choice(candidate_indices)

        # Re-acquire lock to place the algorithm
        with self._lock:
            # Ensure index is still valid
            if chosen_island_index < len(self.islands):
                target_island = self.islands[chosen_island_index]
                target_island.register_algo(algo)
                self.restart_database()

    def restart_database(self):
        """Resets the weaker half of islands every 3600 seconds."""
        # If a max capacity is set, we skip database restart
        if self.algo_database_config.island_capacity is not None:
            return

        with self._lock:
            if time.time() - self._last_reset_time < 3600:
                return
            self._last_reset_time = time.time()

            # Calculate the best score for each island
            island_scores = []
            for island in self.islands:
                if not island.algorithms:
                    island_scores.append(-float("inf"))
                else:
                    # Assuming higher score is better
                    island_scores.append(max(algo.score for algo in island.algorithms))

            island_scores = np.array(island_scores)

            # Sort islands by score. Add noise to break ties randomly.
            indices_sorted_by_score = np.argsort(
                island_scores + np.random.randn(len(island_scores)) * 1e-6
            )

            num_islands = len(self.islands)
            num_islands_to_reset = num_islands // 2

            reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
            keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

            for island_id in reset_islands_ids:
                # Reset the island
                self.islands[island_id] = AlgoIsland(
                    sim_calculator=self.algo_database_config.algo_sim_calculator,
                    island_capacity=self.algo_database_config.island_capacity,
                    exploitation_intensity=self.algo_database_config.selection_exploitation_intensity,
                )

                # Copy a founder from a surviving island
                if len(keep_islands_ids) > 0:
                    founder_island_id = np.random.choice(keep_islands_ids)
                    founder_island = self.islands[founder_island_id]

                    if founder_island.algorithms:
                        # Pick the best algorithm as founder
                        founder = max(founder_island.algorithms, key=lambda x: x.score)
                        self.islands[island_id].register_algo(founder)

    def select_algos(
        self,
        num_islands: int,
        samples_per_island: int,
    ) -> List[tuple[AlgoProto, int]]:
        """Selects algorithms from the database based on two strategies.

        This method can be used for two scenarios:
        1. Select k algorithms from 1 random island: `select_algos(num_islands=1, samples_per_island=k)`
        2. Select 1 algorithm from k random islands: `select_algos(num_islands=k, samples_per_island=1)`

        This method is thread-safe.

        Args:
            num_islands: The number of different islands to sample from.
            samples_per_island: The number of algorithms to sample from each island.

        Returns:
            A list of tuples, where each tuple contains an algorithm prototype
            and the ID of the island it was selected from.
        """
        with self._lock:
            non_empty_islands_with_indices = [
                (i, island) for i, island in enumerate(self.islands) if island
            ]
            if not non_empty_islands_with_indices:
                return []

            # If k is too large, use all non-empty islands
            num_to_select = min(num_islands, len(non_empty_islands_with_indices))
            selected_islands_with_indices = random.sample(
                non_empty_islands_with_indices, num_to_select
            )

            all_selected_algos = []
            for island_id, island in selected_islands_with_indices:
                selected_algos = island.selection(k=samples_per_island)  # Todo add
                for algo in selected_algos:
                    all_selected_algos.append((algo, island_id))

            return all_selected_algos

    def get_island_stats(self) -> Dict[int, int]:
        """Returns a dictionary with the number of algorithms in each island."""
        with self._lock:
            return {i: len(island) for i, island in enumerate(self.islands)}

    def get_total_algo_count(self) -> int:
        """Returns the total number of algorithms across all islands."""
        with self._lock:
            return sum(len(island) for island in self.islands)

    def to_dict(self) -> dict[str, Any]:
        """Converts the AlgoDatabase to a dictionary."""
        return {"islands": [island.to_dict() for island in self.islands]}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: AlgoDatabaseConfig,
    ) -> "AlgoDatabase":
        """Creates an AlgoDatabase from a dictionary."""
        islands = [
            AlgoIsland.from_dict(island_data, config)
            for island_data in data.get("islands", [])
        ]

        return cls(config, islands=islands)


class AlgoIsland:
    def __init__(
        self,
        sim_calculator: BehaveSimCalculator,
        island_capacity: Optional[int],
        exploitation_intensity: float,
    ):
        self.sim_calculator = sim_calculator
        self.island_capacity = island_capacity
        self.algorithms: List[AlgoProto] = []
        self.all_algos: List[AlgoProto] = []  # Store all algorithms ever registered
        self.exploitation_intensity = exploitation_intensity
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.algorithms)

    def calculate_avg_sim_to_island(
        self,
        algo: AlgoProto,
    ) -> tuple[float, dict[str, float]]:
        """Calculates the average similarity from an algorithm to this island.

        This method is thread-safe.

        Args:
            algo: The algorithm to calculate the similarity from.

        Returns:
            A tuple containing:
            - float: The average similarity. Returns 0.0 if the island is empty.
            - dict: Aggregated timings for similarity calculations.
        """
        with self._lock:
            if not self.algorithms:
                return 0.0, {}

            similarities = []
            total_timings = {}
            for island_algo in self.algorithms:
                try:
                    sim, timings = self.sim_calculator.calculate_sim(
                        algo,
                        island_algo,
                    )
                    if np.isnan(sim) or np.isinf(sim):
                        sim = 0.0

                    # Aggregate timings
                    for k, v in timings.items():
                        total_timings[k] = total_timings.get(k, 0.0) + v

                except Exception:
                    sim = 0.0
                similarities.append(sim)

            return float(np.mean(similarities)), total_timings

    def survival(self):
        """Implements a naive survival mechanism for the island.

        If the number of algorithms in the island exceeds `island_capacity`,
        only the algorithms with the highest scores are preserved, up to
        the `island_capacity` limit.
        """
        with self._lock:
            if self.island_capacity is None:
                return
            if len(self.algorithms) > self.island_capacity:
                # Sort algorithms by score in descending order and keep the top ones
                self.algorithms.sort(key=lambda algo: algo.score, reverse=True)
                self.algorithms = self.algorithms[: self.island_capacity]

    def selection(self, k: int = 1) -> List[AlgoProto]:
        """Selects k algorithms from the island using a two-stage process.

        First, a rank-based selection is performed on the unique scores available
        in the island to choose `k` score tiers. Second, for each chosen score,
        a second rank-based selection is performed on the algorithms in that
        tier, where shorter programs are given a higher probability.

        This method is thread-safe.

        Args:
            k: The number of algorithms to select. Defaults to 1.

        Returns:
            A list of `k` selected AlgoProto instances. If the island is empty,
            an empty list is returned.
        """
        with self._lock:
            if not self.algorithms:
                return []

            # 1. Group algorithms by score
            algos_by_score = defaultdict(list)
            for algo in self.algorithms:
                algos_by_score[algo.score].append(algo)

            # 2. Prepare for rank-based selection on the scores
            unique_scores = sorted(list(algos_by_score.keys()), reverse=True)
            if not unique_scores:
                return []

            # Ensure k is not greater than the number of unique scores
            num_to_select = min(k, len(unique_scores))

            # 3. Perform rank-based selection to pick k score tiers
            selected_scores = _rank_based_selection(
                scores=unique_scores,
                choices=unique_scores,
                k=num_to_select,
                exploitation_intensity=self.exploitation_intensity,
            )

            # 4. For each selected score, perform a second rank-based selection
            #    based on code length.
            selected_algos = []
            for score in selected_scores:
                candidate_algos = algos_by_score[score]
                if not candidate_algos:
                    continue

                if len(candidate_algos) == 1:
                    selected_algos.append(candidate_algos[0])
                else:
                    # Higher score for shorter programs (-line_count)
                    line_count_scores = [
                        -len(str(algo.program).split("\n")) for algo in candidate_algos
                    ]
                    # Perform rank-based selection on length
                    picked_algo = _rank_based_selection(
                        scores=line_count_scores,
                        choices=candidate_algos,
                        k=1,
                        exploitation_intensity=0.5,
                    )[0]
                    selected_algos.append(picked_algo)

            return selected_algos

    def register_algo(self, algo: AlgoProto):
        """Registers a new algorithm in the island.

        The new algorithm is added to the island's collection.

        This method is thread-safe.

        Args:
            algo: The AlgoProto instance to register.
        """
        with self._lock:
            self.algorithms.append(algo)
            self.all_algos.append(algo)  # Add to history

            self.survival()

    def get_all_algorithms(self) -> List[AlgoProto]:
        """Returns a list of all algorithms in the island.

        This method is thread-safe.
        """
        with self._lock:
            return list(self.algorithms)

    def to_dict(self) -> dict[str, Any]:
        """Converts the AlgoIsland to a dictionary."""
        return {
            "algorithms": [algo.to_dict() for algo in self.algorithms],
            "all_algos": [algo.to_dict() for algo in self.all_algos],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        algo_database_config: AlgoDatabaseConfig,
    ) -> "AlgoIsland":
        """Creates an AlgoIsland from a dictionary."""
        island = cls(
            sim_calculator=algo_database_config.algo_sim_calculator,
            island_capacity=algo_database_config.island_capacity,
            exploitation_intensity=algo_database_config.selection_exploitation_intensity,
        )
        island.algorithms = [
            AlgoProto.from_dict(algo_data) for algo_data in data["algorithms"]
        ]

        # Restore all_algos
        if "all_algos" in data:
            island.all_algos = [
                AlgoProto.from_dict(algo_data) for algo_data in data["all_algos"]
            ]
        else:
            # Fallback for old data: assume current algorithms are the history
            island.all_algos = list(island.algorithms)

        return island

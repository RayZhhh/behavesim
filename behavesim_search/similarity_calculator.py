import time
from typing import Any, List

from behavesim_search.algo_proto import AlgoProto
import numpy as np


class BehaveSimCalculator:
    """Calculates the similarity between two AlgoProto instances."""

    def __init__(
        self,
        use_dtw=True,
        allow_inequal_len=True,
        traj_sample_k=None,
        traj_trunc_n=None,
    ):
        self.use_dtw = use_dtw
        self.allow_inequal_len = allow_inequal_len
        self.traj_sample_k = traj_sample_k
        self.traj_trunc_n = traj_trunc_n

    def calculate_sim(
        self,
        algo1: AlgoProto,
        algo2: AlgoProto,
    ) -> tuple[float, dict[str, float]]:
        """Calculates the total weighted similarity between two AlgoProto instances.

        Args:
            algo1: The first AlgoProto instance.
            algo2: The second AlgoProto instance.

        Returns:
            A tuple containing:
            - float: The behavioral similarity score between algo1 and algo2.
            - dict: A dictionary containing the execution time (in seconds) for each similarity component.
        """
        timings = {}
        start_time = time.time()
        score = self._behavioral_similarity(algo1, algo2)
        end_time = time.time()
        timings["behavioral"] = end_time - start_time
        total_sim = score
        return total_sim, timings

    def _behavioral_similarity(self, algo1: AlgoProto, algo2: AlgoProto) -> float:
        """Placeholder for calculating behavioral similarity.

        Args:
            algo1: The first AlgoProto instance.
            algo2: The second AlgoProto instance.

        Returns:
            Currently returns 0.0, as this component is not yet implemented.
        """
        sim_on_all_instances = []
        algo1_all_instances_behave, algo2_all_instances_behave = (
            algo1.behavior,
            algo2.behavior,
        )
        algo1_all_instances_behave: List[List[Any]]
        algo2_all_instances_behave: List[List[Any]]

        for i, (pstraj1, pstraj2) in enumerate(
            zip(algo1_all_instances_behave, algo2_all_instances_behave)
        ):
            try:
                sim = self._pstraj_sim(pstraj1, pstraj2)
                sim_on_all_instances.append(sim)
            except:
                sim_on_all_instances.append(0)

        return np.mean(sim_on_all_instances).item()

    def _pstraj_sim(
        self,
        pstraj1: List[Any],
        pstraj2: List[Any],
    ) -> float:
        """Calculate similarity between two problem-solving trajectories (PSTrajs)."""
        pstraj1, pstraj2 = self._sample_points(pstraj1), self._sample_points(pstraj2)
        pstraj1, pstraj2 = self._truncate_traj(pstraj1), self._truncate_traj(pstraj2)

        if not self.allow_inequal_len or not self.use_dtw:
            min_len = min(len(pstraj1), len(pstraj2))
            pstraj1, pstraj2 = pstraj1[:min_len], pstraj2[:min_len]

        if self.use_dtw:
            return 1 - self._cal_dtw_distance(pstraj1, pstraj2)
        else:
            return 1 - self._cal_mean_pairwise_distance(pstraj1, pstraj2)

    def _cal_dtw_distance(self, pstraj1: List[Any], pstraj2: List[Any]) -> float:
        n = len(pstraj1)
        m = len(pstraj2)
        dtw_matrix = np.full((n + 1, m + 1), float("inf"))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self.pairwise_distance_with_norm(pstraj1[i - 1], pstraj2[j - 1])
                last_min = np.min(
                    [
                        dtw_matrix[i - 1, j],  # insertion
                        dtw_matrix[i, j - 1],  # deletion
                        dtw_matrix[i - 1, j - 1],  # match
                    ]
                )
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n, m]

    def _cal_mean_pairwise_distance(
        self, pstraj1: List[Any], pstraj2: List[Any]
    ) -> float:
        dist = []
        for solution1, solution2 in zip(pstraj1, pstraj2):
            dist.append(self.pairwise_distance_with_norm(solution1, solution2))
        return np.mean(dist).item()

    def _sample_points(self, pstraj: List[Any]) -> List[Any]:
        """Sample a point for every k points in the trajectory."""
        if self.traj_sample_k is None or self.traj_sample_k <= 0:
            return pstraj
        return pstraj[:: self.traj_sample_k]

    def _truncate_traj(self, pstraj: List[Any]) -> List[Any]:
        """Truncate the trajectory to the first n points."""
        if self.traj_trunc_n is None or self.traj_trunc_n <= 0:
            return pstraj
        return pstraj[: self.traj_trunc_n]

    def pairwise_distance_with_norm(
        self, solution1: List[int], solution2: List[int]
    ) -> float:
        import editdistance

        return editdistance.distance(solution1, solution2) / max(
            len(solution1), len(solution2)
        )

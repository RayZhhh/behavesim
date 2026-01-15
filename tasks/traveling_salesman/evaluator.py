from typing import Any, Optional, Callable, Dict, List
import numpy as np

from behavesim_search.evaluator import BehaveSimSearchEvaluator, EvalResult
from tasks.traveling_salesman.get_instance import TSPInstanceGenerator


class Evaluator(BehaveSimSearchEvaluator):
    """Evaluator for Traveling Salesman Problem (TSP)."""

    def __init__(self, n_instance=16, problem_size=50):
        """Initialize the TSP evaluator.

        Args:
            n_instance: Number of TSP instances to evaluate on.
            problem_size: Number of cities in each instance.
        """
        super().__init__()
        self.n_instance = n_instance
        self.problem_size = problem_size

        # Use the local generator to create instances
        generator = TSPInstanceGenerator(self.n_instance, self.problem_size)
        self._datasets = generator.generate_instances()

    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs,
    ) -> EvalResult:
        algo_callable = callable_functions_dict["select_next_node"]
        evaluation_result = self.evaluate_(algo_callable)
        if evaluation_result is None:
            # Handle the case where evaluation fails
            return EvalResult(score=-np.inf, behavior=[])
        score, behavior = evaluation_result
        return EvalResult(score=score, behavior=behavior)

    def tour_cost(self, instance: np.ndarray, solution: np.ndarray) -> float:
        """Calculate the total cost (Euclidean distance) of the tour.

        Args:
            instance (np.ndarray): Coordinates of the cities (N, 2).
            solution (np.ndarray): Permutation of city indices representing the tour (N,).

        Returns:
            float: The total distance of the tour.
        """
        # Get coordinates in the order of the solution
        ordered_cities = instance[solution]

        # Calculate vector differences between consecutive cities
        # Rolling shifts the array so we can compute dist(city_i, city_{i+1})
        # The last city will connect back to the first one
        next_cities = np.roll(ordered_cities, -1, axis=0)

        # Compute Euclidean distances
        distances = np.linalg.norm(ordered_cities - next_cities, axis=1)

        return np.sum(distances)

    def generate_neighborhood_matrix(self, instance: np.ndarray) -> np.ndarray:
        """Generate a matrix where each row i contains indices of cities sorted by distance to city i.
        Args:
            instance (np.ndarray): Coordinates of cities (N, 2).

        Returns:
            np.ndarray: Neighborhood matrix (N, N).
        """
        # Efficiently compute pairwise distances using broadcasting
        # diff shape: (N, N, 2)
        diff = instance[:, np.newaxis, :] - instance[np.newaxis, :, :]
        # dist shape: (N, N)
        dist = np.linalg.norm(diff, axis=-1)
        # Argsort to get indices of neighbors sorted by distance for each city
        neighborhood_matrix = np.argsort(dist, axis=1)
        return neighborhood_matrix

    def evaluate_(self, algo_callable: Callable) -> Optional[tuple[float, list]]:
        """Evaluate the heuristic function 'eva' on the datasets.
        Args:
            algo_callable (callable): The heuristic function to decide the next node.
                            Signature: (current_node, destination_node, unvisited_near_nodes, distance_matrix) -> next_node

        Returns:
            A tuple containing:
            - float: The negative average tour distance (higher is better).
            - list: A list of behaviors for each instance.
            Returns None if invalid.
        """
        total_distance = 0.0
        n_ins = 0
        all_behaviors = []

        for instance, distance_matrix in self._datasets:
            # Precompute neighborhood matrix for the current instance
            neighbor_matrix = self.generate_neighborhood_matrix(instance)
            instance_behavior = []

            # Initialize tour state
            current_node = 0
            destination_node = 0  # Convention: start/end at node 0

            route = np.zeros(self.problem_size, dtype=int)
            visited = np.zeros(self.problem_size, dtype=bool)

            route[0] = current_node
            visited[current_node] = True
            instance_behavior.append(route[:1].tolist())

            # Construct the tour step by step
            # We loop up to problem_size - 1 because the last node is determined automatically
            for i in range(1, self.problem_size - 1):
                # Get neighbors of the current node, sorted by distance
                sorted_neighbors = neighbor_matrix[current_node]

                # Filter out visited nodes to get unvisited near nodes
                # Using boolean mask is faster than set operations
                unvisited_mask = ~visited[sorted_neighbors]
                unvisited_near_nodes = sorted_neighbors[unvisited_mask]

                # Execute the heuristic function
                try:
                    next_node = algo_callable(
                        current_node,
                        destination_node,
                        unvisited_near_nodes,
                        distance_matrix,
                    )
                except Exception:
                    return None  # Function execution failed

                # Validate the output
                if not isinstance(next_node, (int, np.integer)):
                    try:
                        next_node = int(next_node)
                    except:
                        return None

                if (
                    next_node < 0
                    or next_node >= self.problem_size
                    or visited[next_node]
                ):
                    return None  # Invalid node selected

                # Update state
                visited[next_node] = True
                current_node = next_node
                route[i] = current_node
                instance_behavior.append(route[: i + 1].tolist())

            # Handle the last remaining node
            last_node_indices = np.where(~visited)[0]
            if len(last_node_indices) != 1:
                return None  # Should not happen

            current_node = last_node_indices[0]
            route[self.problem_size - 1] = current_node
            instance_behavior.append(route.tolist())
            all_behaviors.append(instance_behavior)

            # Calculate total tour cost
            dist = self.tour_cost(instance, route)
            total_distance += dist

            n_ins += 1
            if n_ins >= self.n_instance:
                break

        ave_dis = total_distance / n_ins
        return -ave_dis, all_behaviors


if __name__ == "__main__":
    import sys

    code = '''
import numpy as np

def select_next_node(
    current_node: int,
    destination_node: int,
    unvisited_nodes: np.ndarray,
    distance_matrix: np.ndarray,
) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        destination_node: ID of the destination node.
        unvisited_nodes: Array of IDs of unvisited nodes.
        distance_matrix: Distance matrix of nodes.

    Return:
        ID of the next node to visit.
    """
    # Greedy strategy: choose the unvisited node closest to the current node
    # (Original example logic was closest to destination, but unvisited_nodes are passed in unknown order?
    # No, unvisited_nodes are sorted by distance to current_node in the evaluator.
    # But here we access distance_matrix[current_node][unvisited_nodes])
    distances_to_destination = distance_matrix[current_node][unvisited_nodes]
    # Find the index of the unvisited node with the smallest distance
    next_node_index = np.argmin(distances_to_destination)
    # Get the ID of the next node to visit
    next_node = unvisited_nodes[next_node_index]
    return next_node
'''
    tsp = Evaluator()
    res = tsp.secure_evaluate(code)
    print(res["result"]["behavior"][0][:5])

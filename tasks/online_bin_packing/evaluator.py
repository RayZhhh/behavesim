from typing import Any, Callable, Dict, List
import numpy as np

from behavesim_search.evaluator import BehaveSimSearchEvaluator, EvalResult
from tasks.online_bin_packing.generate_weibull_instances import generate_weibull_dataset


class Evaluator(BehaveSimSearchEvaluator):
    """Evaluator for online bin packing problem."""

    def __init__(self, n_instances=5, n_items=5000, capacity=100, **kwargs):
        """
        Args:
            - 'data_file' (str): The data file to load (default is 'weibull_5k_train.pkl').
            - 'data_key' (str): The key of the data to load (default is 'data_key').

        Raises:
            AttributeError: If the data key does not exist.
            FileNotFoundError: If the specified data file is not found.
        """
        super().__init__()
        self.n_instances = n_instances
        self.n_items = n_items
        self.capacity = capacity
        self._datasets = generate_weibull_dataset(
            self.n_instances, self.n_items, self.capacity
        )

    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs,
    ) -> EvalResult:
        if not callable_functions_dict or "priority" not in callable_functions_dict:
            return EvalResult(score=-np.inf, behavior=[])

        priority_func = callable_functions_dict["priority"]
        score, behavior = self.evaluate_heuristic(priority_func)
        return EvalResult(score=score, behavior=behavior)

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]

    def online_binpack(
        self, items: tuple[float, ...], bins: np.ndarray, priority: Callable
    ):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        
        behavior_sequence = []
        decision_sequence = []

        # Add items to bins.
        for i, item in enumerate(items):
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            if len(valid_bin_indices) == 0:
                continue
            # Score each bin based on heuristic.
            priorities = priority(item, bins[valid_bin_indices])
            # Add item to bin with the highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)

            if i < 100:
                decision_sequence.append(int(best_bin))
                behavior_sequence.append(decision_sequence[:])

        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins, behavior_sequence

    def evaluate_heuristic(self, priority: Callable) -> tuple[float, list]:
        """Evaluate heuristic function on a set of online binpacking instances."""
        # List storing number of bins used for each instance.
        num_bins = []
        all_behaviors = []
        # Perform online binpacking for each instance.
        for name in self._datasets:
            instance = self._datasets[name]
            capacity = instance["capacity"]
            items = instance["items"]
            # Create num_items bins so there will always be space for all items,
            # regardless of packing order. Array has shape (num_items,).
            bins = np.array([capacity for _ in range(instance["num_items"])])
            # Pack items into bins and return remaining capacity in bins_packed, which
            # has shape (num_items,).
            _, bins_packed, behavior = self.online_binpack(items, bins, priority)
            all_behaviors.append(behavior)

            # If remaining capacity in a bin is equal to initial capacity, then it is
            # unused. Count number of used bins.
            num_bins.append((bins_packed != capacity).sum())
        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).
        score = -np.mean(num_bins).item()
        return score, all_behaviors


if __name__ == "__main__":
    priority_program = """
import numpy as np

def priority(item: float, valid_bins: np.ndarray) -> np.ndarray:
    '''
    Priority function for the First-Fit Decreasing (FFD) heuristic.

    Args:
        item: The size of the item to be packed.
        valid_bins: A numpy array of remaining capacities in valid bins.

    Returns:
        A numpy array of priorities for the valid bins.
    '''
    return -(valid_bins - item)
"""

    obp = Evaluator()
    results = obp.evaluate(priority_program)
    print(f"Evaluation result: {results['result']}")

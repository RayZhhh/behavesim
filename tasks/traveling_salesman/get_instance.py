import numpy as np
from typing import List, Tuple


class TSPInstanceGenerator:
    """Generator for Traveling Salesman Problem (TSP) instances."""

    def __init__(self, n_instances: int, n_cities: int, seed: int = 2024):
        """
        Initialize the TSP instance generator.

        Args:
            n_instances (int): Number of instances to generate.
            n_cities (int): Number of cities (nodes) in each instance.
            seed (int): Random seed for reproducibility. Default is 2024.
        """
        self.n_instances = n_instances
        self.n_cities = n_cities
        self.seed = seed

    def generate_instances(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate TSP instances with random coordinates and corresponding distance matrices.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples, where each tuple contains:
                - coordinates (np.ndarray): Shape (n_cities, 2), random coordinates of cities in [0, 1].
                - distances (np.ndarray): Shape (n_cities, n_cities), Euclidean distance matrix between cities.
        """
        np.random.seed(self.seed)
        instance_data = []
        for _ in range(self.n_instances):
            # Generate random coordinates in [0, 1]
            coordinates = np.random.rand(self.n_cities, 2)
            # Calculate Euclidean distance matrix efficiently using broadcasting
            # diff shape: (n_cities, n_cities, 2)
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            # distances shape: (n_cities, n_cities)
            distances = np.linalg.norm(diff, axis=-1)
            instance_data.append((coordinates, distances))

        return instance_data

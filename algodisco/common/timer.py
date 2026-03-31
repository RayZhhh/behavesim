# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import time
from typing import Any
from algodisco.base.algo import AlgoProto


class Timer:
    """
    A context manager to measure the execution time of a code block
    and store it in an AlgoProto instance's attributes.

    Usage:
        algo = AlgoProto(...)
        with Timer(algo, "evaluation_time"):
            # Your code here
            time.sleep(0.1)
        print(algo["evaluation_time"]) # Access the recorded time
    """

    def __init__(self, algo_proto_instance: AlgoProto, key: str):
        """
        Initializes the Timer context manager.

        Args:
            algo_proto_instance: The AlgoProto instance to store the time in.
            key: The key under which the duration will be stored in algo_proto.attributes.
        """
        if not isinstance(algo_proto_instance, AlgoProto):
            raise TypeError("algo_proto_instance must be an instance of AlgoProto")
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        self.algo_proto_instance = algo_proto_instance
        self.key = key
        self.start_time = None

    def __enter__(self):
        """Starts the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """
        Stops the timer, calculates the duration, and stores it.

        Args:
            exc_type: Exception type if an exception occurred within the 'with' block.
            exc_val: Exception value if an exception occurred within the 'with' block.
            exc_tb: Traceback if an exception occurred within the 'with' block.
        """
        end_time = time.time()
        duration = end_time - self.start_time
        # Use AlgoProto's __setitem__ to store in attributes
        self.algo_proto_instance[self.key] = duration
        # Exceptions are re-raised by default by __exit__ if it returns None (which it does here)


# A convenience function to instantiate the Timer
def timer(algo_proto_instance: AlgoProto, key: str) -> Timer:
    """
    Convenience function to create a Timer context manager.

    Args:
        algo_proto_instance: The AlgoProto instance to store the time in.
        key: The key under which the duration will be stored in algo_proto.attributes.

    Returns:
        An instance of the Timer context manager.
    """
    return Timer(algo_proto_instance, key)

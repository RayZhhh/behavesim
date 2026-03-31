# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import copy
import functools
from typing import Literal, Optional, Any, Callable, Dict

from algodisco.toolkit.sandbox.sandbox_executor import (
    SandboxExecutor,
    ExecutionResults,
)
from algodisco.toolkit.sandbox.sandbox_executor_simple import SandboxExecutorSimple
from algodisco.toolkit.sandbox.sandbox_executor_ray import SandboxExecutorRay

__all__ = ["sandbox_run"]


def _is_class_method(func) -> bool:
    return "." in func.__qualname__ and "<locals>" not in func.__qualname__


def _is_top_level_function(func) -> bool:
    return func.__qualname__ == func.__name__


class _FunctionWorker:
    """Helper class to wrap standalone functions for SandboxExecutor."""

    def __init__(self, func):
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def sandbox_run(
    sandbox_type: Literal["ray", "process", "simple"] = "simple",
    timeout: Optional[float] = None,
    redirect_to_devnull: bool = False,
    ray_actor_options: Optional[Dict[str, Any]] = None,
    add_execution_time_in_res_dict: bool = True,
    add_error_msg_in_res_dict: bool = True,
    **executor_init_kwargs,
):
    """
    Decorator to execute a class method or standalone function in a sandbox (Process, Simple, or Ray).

    When a method/function decorated with @sandbox_run is called, it will be executed
    in a separate process (or Ray actor).

    Args:
        sandbox_type: The type of sandbox to use. 'ray' for SandboxExecutorRay,
                      'process' for SandboxExecutor (shared memory), 'simple' for
                      SandboxExecutorSimple (Queue-based). Defaults to 'simple'.
        timeout: Timeout in seconds for the execution.
        redirect_to_devnull: Whether to redirect stdout/stderr to /dev/null
                             inside the sandbox.
        ray_actor_options: Options for the Ray actor (only used if sandbox_type='ray').
        **executor_init_kwargs: Additional keyword arguments passed to the
                                Executor's constructor (e.g., debug_mode, init_ray).
    """
    # Followings are to cheat IDE
    executor_init_kwargs.get("debug_mode", False)
    executor_init_kwargs.get("init_ray", None)
    executor_init_kwargs.get("recur_kill_eval_proc", False)

    def decorator(func: Callable) -> Callable:
        is_class_method = _is_class_method(func)  # noqa

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_class_method:
                # Treated as a method call: args[0] is 'self'
                if not args:
                    raise RuntimeError("Method call expected 'self' as first argument.")
                self_instance = args[0]
                method_args = args[1:]

                # Check bypass flag to prevent recursion
                if getattr(self_instance, "_bypass_sandbox", False):
                    return func(self_instance, *method_args, **kwargs)

                # Create worker copy
                evaluate_worker = copy.copy(self_instance)
                evaluate_worker._bypass_sandbox = True
                method_name = func.__name__
            else:
                # Treated as a standalone function
                worker = _FunctionWorker(func)
                evaluate_worker = worker
                method_name = "run"
                method_args = args

            # Prepare Executor
            if sandbox_type == "ray":
                import ray

                init_ray = executor_init_kwargs.get("init_ray", None)
                if init_ray is None:
                    init_ray = not ray.is_initialized()

                executor = SandboxExecutorRay(
                    evaluate_worker,
                    init_ray=init_ray,
                    **executor_init_kwargs,
                )
                ray_options = ray_actor_options
            elif sandbox_type == "simple":
                executor = SandboxExecutorSimple(
                    evaluate_worker, **executor_init_kwargs
                )
                ray_options = None
            else:
                executor = SandboxExecutor(evaluate_worker, **executor_init_kwargs)
                ray_options = None  # Not used for process

            # Execute
            if sandbox_type == "ray":
                result = executor.secure_execute(
                    worker_execute_method_name=method_name,
                    method_args=method_args,
                    method_kwargs=kwargs,
                    timeout_seconds=timeout,
                    redirect_to_devnull=redirect_to_devnull,
                    ray_actor_options=ray_options,
                )
            else:
                result = executor.secure_execute(
                    worker_execute_method_name=method_name,
                    method_args=method_args,
                    method_kwargs=kwargs,
                    timeout_seconds=timeout,
                    redirect_to_devnull=redirect_to_devnull,
                )

            # Handle case when result is None (e.g., timeout or execution failure)
            if result is None:
                return None

            # Create a new dict to hold the results, starting with the actual result if exists
            if result["result"] is not None:
                actual_result = result["result"]
                # If the result is not a dict, wrap it in a dict
                if isinstance(actual_result, dict):
                    final_results = actual_result
                else:
                    final_results = {"result": actual_result}
            else:
                final_results = {}

            if (
                add_execution_time_in_res_dict
                and result.get("execution_time") is not None
            ):
                final_results.update({"execution_time": result["execution_time"]})
            if add_error_msg_in_res_dict and result.get("error_msg") is not None:
                final_results.update({"error_msg": result["error_msg"]})

            return final_results

        return wrapper

    return decorator

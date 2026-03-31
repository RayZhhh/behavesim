# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import logging
from typing import Optional


def format_time_info(
    eval_time_val: float,
    execution_time_val: Optional[float] = None,
) -> str:
    """Format time information for terminal output.

    Args:
        eval_time_val: The total evaluation time (from Timer).
        execution_time_val: The actual execution time from sandbox (optional).

    Returns:
        Formatted time string like "(Eval: 5.00s)" or "(Eval: 5.00s, run: 4.50s)"
    """
    eval_time_str = f"{eval_time_val:6.2f}s"

    time_parts = [f"Eval: {eval_time_str}"]

    if execution_time_val is not None:
        time_parts.append(f"run: {execution_time_val:6.2f}s")

    return " (" + ", ".join(time_parts) + ")"


def format_error_box(error_msg: str, max_width: int = 80) -> str:
    """Format an error message in a visible box.

    Args:
        error_msg: The error message to format.
        max_width: Maximum width of the box (default 80).

    Returns:
        Formatted string with error message in a box.
    """
    # Truncate error message if too long
    if len(error_msg) > max_width - 4:
        error_msg = error_msg[: max_width - 7] + "..."

    lines = error_msg.split("\n")
    if len(lines) > 5:
        lines = lines[:5] + ["..."]

    width = max(len(line) for line in lines) + 4
    width = min(width, max_width)

    box = [
        "",
        "=" * width,
        "| ERROR |".center(width),
        "-" * width,
    ]

    for line in lines:
        box.append(f"| {line:<{width - 2}} |")

    box.append("=" * width)
    box.append("")

    return "\n".join(box)


def log_with_error_box(logger, error_msg: str, level: int = logging.ERROR):
    """Log an error message with a visible box.

    Args:
        logger: The logger instance.
        error_msg: The error message to log.
        level: Logging level (default ERROR).
    """
    formatted = format_error_box(error_msg)
    logger.log(level, formatted)

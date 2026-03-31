# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import argparse
from pathlib import Path

# Automatically detect the project root (4 levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from algodisco.common.config_loading import (
    build_component,
    build_method_config,
    load_yaml_config,
)
from algodisco.methods.eoh.config import EoHConfig
from algodisco.methods.eoh.search import EoHSearch


def main():
    parser = argparse.ArgumentParser(description="Run EoH algorithm search.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "examples" / "online_bin_packing" / "configs" / "eoh.yaml"),
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    config_data = load_yaml_config(args.config)
    method_config, debug_mode, debug_mode_crash = build_method_config(
        config_data=config_data,
        project_root=PROJECT_ROOT,
        config_cls=EoHConfig,
    )

    # Dynamically instantiate components
    llm = build_component(
        section_config=config_data.get("llm", {}),
        project_root=PROJECT_ROOT,
    )
    evaluator = build_component(
        section_config=config_data.get("evaluator", {}),
        project_root=PROJECT_ROOT,
    )
    logger = build_component(
        section_config=config_data.get("logger", {}),
        project_root=PROJECT_ROOT,
        path_kwargs=("logdir", "swanlab_logdir"),
    )

    if not llm:
        raise ValueError("An LLM must be provided in the configuration.")
    if not evaluator:
        raise ValueError("An Evaluator must be provided in the configuration.")

    search = EoHSearch(
        config=method_config, llm=llm, evaluator=evaluator, logger=logger
    )

    # Set debug mode from config
    search.debug_mode = debug_mode
    search.debug_mode_crash = debug_mode_crash

    search.run()


if __name__ == "__main__":
    main()

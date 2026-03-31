# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import dataclasses
import importlib
import importlib.util
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import yaml


def load_class(
    class_path: str | None = None,
    kwargs: dict[str, Any] | None = None,
    project_root: Path | None = None,
):
    """Dynamically imports and instantiates a class from a config section."""
    if not class_path:
        return None

    if kwargs is None:
        kwargs = {}

    if ":" in class_path and class_path.split(":", 1)[0].endswith(".py"):
        file_path_str, class_name = class_path.rsplit(":", 1)
        file_path = Path(file_path_str)
        if not file_path.is_absolute() and project_root is not None:
            file_path = project_root / file_path

        module_name = f"_algodisco_dynamic_{file_path.stem}_{abs(hash(file_path))}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from file path: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)

    cls = getattr(module, class_name)
    return cls(**kwargs)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file into a dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(project_root: Path, path_value: str | None) -> str | None:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


def _resolve_dataclass_type(field_type: Any) -> type | None:
    """Returns the underlying dataclass type for a field annotation, if any."""
    if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
        return field_type

    origin = get_origin(field_type)
    if origin in (list, dict, tuple):
        return None

    if origin is None:
        return None

    if origin not in (Union, UnionType):
        return None

    for arg in get_args(field_type):
        nested = _resolve_dataclass_type(arg)
        if nested is not None:
            return nested
    return None


def instantiate_dataclass_from_dict(
    config_cls: type, config_data: dict[str, Any], project_root: Path
) -> Any:
    """Instantiates a dataclass, recursively resolving nested config blocks."""
    method_config_data = dict(config_data)

    for field in dataclasses.fields(config_cls):
        if field.name not in method_config_data:
            continue

        value = method_config_data[field.name]
        if not isinstance(value, dict):
            continue

        if "class_path" in value:
            method_config_data[field.name] = load_class(
                class_path=value.get("class_path"),
                kwargs=value.get("kwargs", {}),
                project_root=project_root,
            )
            continue

        nested_dataclass = _resolve_dataclass_type(field.type)
        if nested_dataclass is not None:
            method_config_data[field.name] = instantiate_dataclass_from_dict(
                nested_dataclass, value, project_root
            )

    return config_cls(**method_config_data)


def build_method_config(
    config_data: dict[str, Any],
    project_root: Path,
    config_cls: type,
) -> tuple[Any, bool, bool]:
    """Builds a method config dataclass from the YAML dictionary."""
    method_config_data = dict(config_data.get("method", {}))

    debug_mode = method_config_data.pop("debug_mode", False)
    debug_mode_crash = method_config_data.pop("debug_mode_crash", False)

    if "template_program_path" in method_config_data:
        template_path = Path(
            _resolve_path(project_root, method_config_data.pop("template_program_path"))
        )
        with open(template_path, "r") as f:
            method_config_data["template_program"] = f.read()

    task_desc_path = method_config_data.get("task_description_path")
    if task_desc_path:
        task_desc_path = Path(
            _resolve_path(project_root, method_config_data.pop("task_description_path"))
        )
        with open(task_desc_path, "r") as f:
            method_config_data["task_description"] = f.read()
    elif "task_description_path" in method_config_data:
        method_config_data.pop("task_description_path")

    if method_config_data.get("task_description") is None:
        method_config_data["task_description"] = ""

    if "template_dir" in method_config_data and method_config_data["template_dir"]:
        method_config_data["template_dir"] = _resolve_path(
            project_root, method_config_data["template_dir"]
        )

    return (
        instantiate_dataclass_from_dict(config_cls, method_config_data, project_root),
        debug_mode,
        debug_mode_crash,
    )


def build_component(
    section_config: dict[str, Any],
    project_root: Path,
    path_kwargs: tuple[str, ...] = (),
):
    """Instantiates a component from a config section, resolving relative paths."""
    if not section_config:
        return None

    kwargs = dict(section_config.get("kwargs", {}))
    for key in path_kwargs:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = _resolve_path(project_root, kwargs[key])

    return load_class(
        class_path=section_config.get("class_path"),
        kwargs=kwargs,
        project_root=project_root,
    )

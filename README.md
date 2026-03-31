<p align="center">
  <img src="assets/behavesim.png" alt="behavesim"/>
</p>

<h1 align="center">
BehaveSim: Rethinking Code Similarity for<br/>Automated Algorithm Design with LLMs
</h1>

<p align="center">
  Code for the ICLR 2026 paper
  <a href="https://openreview.net/pdf?id=HIUqeO9OOr">
    Rethinking Code Similarity for Automated Algorithm Design with LLMs
  </a>
</p>

<p align="center">
  <a href="https://github.com/RayZhhh/behavesim"><img src="https://img.shields.io/github/stars/RayZhhh/behavesim" alt="Stars"></a>
  <a href="https://github.com/RayZhhh/behavesim"><img src="https://img.shields.io/github/forks/RayZhhh/behavesim" alt="Forks"></a>
  <a href="https://github.com/RayZhhh/behavesim/blob/main/LICENSE"><img src="https://img.shields.io/github/license/RayZhhh/behavesim" alt="License"></a>
  <a href="https://github.com/RayZhhh/behavesim#tutorial"><img src="https://img.shields.io/badge/Usage_Tutorial-BehaveSim-1F6FEB" alt="Usage Tutorial"></a>
  <a href="https://github.com/RayZhhh/algodisco/blob/main/docs_en/user-guide/search-methods/index.md"><img src="https://img.shields.io/badge/Framework_Guide-AlgoDisco-0A7F5A" alt="Framework Guide"></a>
  <a href="https://deepwiki.com/RayZhhh/algodisco/"><img src="https://img.shields.io/badge/DeepWiki-AlgoDisco-0A7F5A" alt="DeepWiki"></a>
</p>

## 📝 Citation

```bibtex
@inproceedings{zhang2026rethinking,
  title={Rethinking Code Similarity for Automated Algorithm Design with LLMs},
  author={Zhang, Rui and Lu, Zhichao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  url={https://openreview.net/pdf?id=HIUqeO9OOr},
  year={2026}
}
```

## ✨ Highlights

- Paper implementation of BehaveSim for LLM-driven automated algorithm design.
- Includes a modular `algodisco` package layout with `FunSearch` and `FunSearch + BehaveSim`.
- Supports configurable similarity calculators, multi-island search, and optional SwanLab logging.
- Provides a short Python entrypoint and YAML-based method configs for quick experimentation.

## 📌 Repository Scope

This repository is a focused, lightweight companion repo for the BehaveSim paper.
It keeps the runnable method code and minimal usage entrypoints here, but intentionally
does not duplicate the full framework tutorial, architecture walkthrough, or broader
method documentation.

If you want to understand the overall search framework design, method abstractions,
configuration philosophy, and end-to-end workflow, please use
[AlgoDisco](https://github.com/RayZhhh/algodisco) as the primary reference.

## 🛠️ Requirements

- Python >= 3.11
- Python 3.12 is recommended

## 🚀 Installation

First, clone the repository:

```bash
git clone https://github.com/RayZhhh/behavesim.git
cd behavesim
```

### Option 1: `uv` (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip sync
```

### Option 2: `pip`

```bash
pip install -e .
```

## 🚀 Quick Start

The shortest path is to run the provided Python example:

1. Open `run_search.py`.
2. Replace the `Todo` placeholders for your `base_url`, `api_key`, logger settings, and parallelism values.
3. Run one of the following commands:

```bash
uv run run_search.py
python run_search.py
nohup uv run run_search.py > log.out 2>&1 &
nohup python run_search.py > log.out 2>&1 &
```

If you prefer the YAML entrypoint, run:

```bash
python -m algodisco.methods.funsearch_behavesim.main_funsearch_behavesim --config path/to/your_config.yaml
```

Starter config templates are provided in:

- `algodisco/methods/funsearch_behavesim/configs/run_pickle_logger.yaml`
- `algodisco/methods/funsearch_behavesim/configs/run_swanlab_logger.yaml`

## 📖 Tutorial

For repository users, the recommended reading path is:

1. [Read AlgoDisco's quick start](https://github.com/RayZhhh/algodisco/blob/main/docs_en/getting-started/quickstart.md) for the overall workflow.
2. [Read AlgoDisco's search-method documentation](https://github.com/RayZhhh/algodisco/blob/main/docs_en/user-guide/search-methods/index.md) for the framework-level method design.
3. [Open this repository's runnable example](https://github.com/RayZhhh/behavesim/blob/main/run_search.py) for the minimum BehaveSim-specific setup.
4. [Inspect this repository's BehaveSim YAML templates](https://github.com/RayZhhh/behavesim/tree/main/algodisco/methods/funsearch_behavesim/configs) for the method-specific config fields.

In short: use this repository for the paper-specific code, and use
[AlgoDisco](https://github.com/RayZhhh/algodisco) for the fuller tutorial and architecture context.

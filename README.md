# Code implementation for "Rethinking Code Similarity for Automated Algorithm Design with LLMs"

## Installation

First, clone the repository:
```bash
git clone https://github.com/RayZhhh/behavesim.git
cd behavesim
```

> **Important:** This project requires Python >=3.10 (3.11 is recommended).

There are two ways to set up the environment:

### 1. Using `uv` (Recommended)

This workflow creates a virtual environment and installs all dependencies from the `pyproject.toml` and `uv.lock` files.

```bash
# Create and activate the virtual environment
uv venv
source .venv/bin/activate

# Sync the environment to match the lock file
uv pip sync
```

### 2. Using `pip`

This method assumes you have a Python environment ready. You may need to generate a `requirements.txt` first or install packages directly.

```bash
pip install py-adtools numpy scipy swanlab black editdistance
```

### 3. Run `FunSearch+BehaveSim`

Please first check `Todo` items in `run_search.py`. Run the script using any of the following commands: 

```bash
uv run run_search.py
python run_search.py
nohup uv run run_search.py > log.out 2>&1 &
nohup python run_search.py > log.out 2>&1 &
```

import os
from pathlib import Path

PROJECT_NAME = "DABenchmark"
DESCRIPTION = """
Distilling a model through an API.
"""

PROJECT_DIR = Path.home() / PROJECT_NAME
DATA_DIR = PROJECT_DIR / "data"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
ARGS_DIR = PROJECT_DIR / "args"
SWEEPS_DIR = PROJECT_DIR / "sweeps"
UTILS_DIR = PROJECT_DIR / "utils"


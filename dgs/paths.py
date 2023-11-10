"""
basic paths and folder structure to be loaded elsewhere
"""

import os

from pathlib import Path

# project paths
config_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = Path(config_path).parent

# system paths
CONFIGS_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"
TEST_DIR = PROJECT_DIR / "tests"

# module paths
DEPENDENCIES_DIR = PROJECT_DIR / "dependencies"
ALPHA_POSE_DIR = DEPENDENCIES_DIR / "AlphaPose_Fork"
ALPHA_POSE_CONFIG_DIR = ALPHA_POSE_DIR / "configs"
TORCHREID_DIR = DEPENDENCIES_DIR / "torchreid"

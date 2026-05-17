from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

HYDRA_CONFIG_PATH = PACKAGE_ROOT / "configs"

RESOURCE_ROOT = PACKAGE_ROOT / "resources"
BENCHMARKS_ROOT = RESOURCE_ROOT / "benchmarks"
BENCHMARK_PDB_DIR = BENCHMARKS_ROOT / "pdb"

PACKAGE_PATH_PREFIX = "package://"
PACKAGE_RESOURCE_ROOT = RESOURCE_ROOT
PROJECT_PATH = PACKAGE_RESOURCE_ROOT

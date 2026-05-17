from __future__ import annotations

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from dynamicmpnn.constants import REPO_ROOT
from dynamicmpnn.eval.pipeline import run_evaluation


load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(REPO_ROOT / ".env.local", override=False)


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="evaluate",
)
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()  # type: ignore[misc]

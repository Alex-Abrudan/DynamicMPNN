"""The released single-chain checkpoints load with the default pooling, and
that is correct even though it looks wrong.

`single_chain_k{2,3,5}.ckpt` store no `pooling_strategy`, so they load with the
constructor default `'all_chains_equal'` rather than `'single_chain_k'`. That
reads like a bug, and the obvious fixes -- forcing the intended strategy, or
refusing the checkpoints -- are both regressions:

* Measured over 16 benchmark targets at 8 samples, mean sequence recovery is
  33.9% (k=2), 37.1% (k=3) and 37.5% (k=5), against 41.9% for the multi-chain
  model and 5.0% for chance. That is the "moderate albeit expected drop"
  Table 3 of the paper reports for k=2 single-chain versus multi-chain. These
  are working models.
* Forcing `pooling_strategy='single_chain_k'` raises inside the model
  (`normalized_shape=[26] ... got input of size[143]`), because the evaluation
  pipeline builds two-state pair graphs with context chains rather than the k
  single chains that path expects.

So the default route is the working one and the current behaviour is
load-bearing. This test is the tripwire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from dynamicmpnn.eval import sampling  # noqa: E402

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
PASSTHROUGH_CFG = OmegaConf.create({"model": {"temperature": None,
                                              "refresh_interval": None}})


@pytest.mark.parametrize("name", ["single_chain_k2.ckpt", "single_chain_k3.ckpt",
                                  "single_chain_k5.ckpt"])
def test_single_chain_checkpoints_keep_the_default_pooling(name):
    path = CHECKPOINTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")

    model = sampling._load_model_for_sampling(path, PASSTHROUGH_CFG, num_samples=1)

    assert model.pooling_strategy == "all_chains_equal", (
        f"{name} now loads as {model.pooling_strategy!r}. If that was "
        "deliberate, check decoding still works before keeping it: forcing "
        "'single_chain_k' raises on the two-state pair graphs the evaluation "
        "pipeline builds, and the default route is what produces the ~34-38% "
        "recovery these checkpoints are documented to achieve.")

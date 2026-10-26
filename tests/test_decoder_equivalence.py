"""The fast decoder must reproduce the reference decoder exactly.

`sample_v3` is the reference implementation: it produced the published results
and shares no machinery with `sample_batched`, which is what makes its
agreement meaningful. `sample()` routes to `sample_batched`.

Both paths are teacher-forced to the same sequence so no stochastic draw can
mask a discrepancy, and the comparison is on LOGITS rather than sampled
sequences: floating-point reduction order differs between the two (different
tensor shapes tile differently), which can flip a `Categorical` draw near a
decision boundary. Sequence equality is therefore neither necessary nor
sufficient; logit agreement is the invariant.

Measured agreement across all 96 benchmark targets (N=25..1118) is 1.19e-05
worst case against a float32 epsilon of ~1.2e-07, and sampled sequences were
byte-identical on 96/96. The tolerance below is deliberately looser so the test
does not fail on hardware whose reduction order differs again.

SPEED: this is marked `slow` because a decode costs ~100s on CPU (seconds on a
GPU). Run it with `pytest -m slow`, or everything-but with
`pytest -m "not slow"`. Setup is shared across tests via a module-scoped
fixture, since loading the checkpoint and featurising dominate otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("graphein")

from dynamicmpnn.eval import sampling  # noqa: E402
from dynamicmpnn.eval.inputs import resolve_inputs  # noqa: E402
from dynamicmpnn.eval.pipeline import build_request  # noqa: E402

# Smallest bundled multi-state target (25 designed residues), which still
# exercises several periodic-refresh cycles at refresh_interval=4.
TARGET = "2mwf_2nnt"
CHECKPOINT = "checkpoints/multi_chain_reload.ckpt"
NUM_SAMPLES = 1
TOLERANCE = 1e-3

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def decoder_fixture(tmp_path_factory):
    """Build the model and graph once; each build costs a checkpoint load."""
    from hydra import compose, initialize_config_dir
    from torch_geometric.data import Batch

    from dynamicmpnn import configs

    tmp_path = tmp_path_factory.mktemp("decoder_equivalence")
    with initialize_config_dir(version_base="1.3",
                               config_dir=str(Path(configs.__file__).parent)):
        cfg = compose(config_name="evaluate", overrides=[
            f"eval={TARGET}",
            f"eval.model_ref={CHECKPOINT}",
            f"eval.num_samples={NUM_SAMPLES}",
            "eval.af3_evaluate=false",
            f"output_dir={tmp_path}",
        ])

    request = build_request(cfg)
    bundle = resolve_inputs(request)
    checkpoint = torch.load(str(request.resolved_model_path), map_location="cpu",
                            weights_only=False)
    features_cfg = sampling._checkpoint_config(checkpoint).get("features", cfg.features)
    graph = sampling._create_featuriser(features_cfg)(bundle.multi_state_data,
                                                      pdb_code=TARGET)
    model = sampling._load_model_for_sampling(request.resolved_model_path, cfg,
                                              NUM_SAMPLES, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), Batch.from_data_list([graph]).to(device)


@pytest.mark.parametrize("refresh_interval", [0, 4])
def test_batched_decoder_matches_reference(decoder_fixture, refresh_interval):
    """refresh_interval=0 exercises the plain loop, 4 the periodic refresh.

    The refresh uses a different edge set from the per-residue step (only edges
    with BOTH endpoints already decoded), so it needs its own coverage.
    """
    model, batch = decoder_fixture
    num_nodes = int(batch.num_decoder_nodes.sum())
    generator = torch.Generator().manual_seed(1234)
    forced = torch.randint(0, model.out_dim, (num_nodes,),
                           generator=generator).to(batch.seq.device)

    with torch.inference_mode():
        _, reference_logits, _, _ = model.sample_v3(
            batch, return_logits=True, refresh_interval=refresh_interval,
            forced_seq=forced)
        _, batched_logits, _ = model.sample_batched(
            batch, return_logits=True, refresh_interval=refresh_interval,
            forced_seq=forced)

    assert len(batched_logits) == 1
    difference = (reference_logits.float() - batched_logits[0].float()).abs().max()
    assert difference <= TOLERANCE, (
        f"sample_batched diverged from the sample_v3 reference by {difference:.3e} "
        f"at refresh_interval={refresh_interval}; tolerance {TOLERANCE}. "
        "The fast decoder no longer computes the same function."
    )


def test_sample_returns_reference_shapes(decoder_fixture):
    """sample() must keep sample_v3's return contract now that it routes
    through sample_batched, which returns per-graph lists.

    A mistake here corrupts output silently rather than raising, so it is worth
    asserting separately from the numerical equivalence above.
    """
    model, batch = decoder_fixture
    num_nodes = int(batch.num_decoder_nodes.sum())

    with torch.inference_mode():
        sequences, logits, indices, batch_indices = model.sample(
            batch, return_logits=True)

    assert sequences.shape == (NUM_SAMPLES, num_nodes)
    assert logits.shape == (NUM_SAMPLES, num_nodes, model.out_dim)
    assert indices.shape[0] == num_nodes
    assert batch_indices.shape[0] == num_nodes
    assert int(batch_indices.max()) == 0, "single graph => all batch indices zero"

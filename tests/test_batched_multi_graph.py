"""Decoding several graphs together must match decoding them one at a time.

`sample()` routes to `sample_batched`, which concatenates graphs into a single
node space with per-graph offsets and advances them in lockstep. None of that
machinery is exercised by a single-graph batch, yet it is now a live path for
any caller passing more than one graph:

  * per-graph node offsets, and slicing results back apart;
  * the `active` set, so graphs shorter than the longest drop out partway;
  * the refresh exclusion, which must stop refreshing a graph once it has
    finished so its refresh history matches a standalone decode;
  * `sample()`'s reassembly of per-graph lists into `sample_v3`'s concatenated
    return contract -- getting that wrong corrupts output silently.

Targets differ in length on purpose (N = 25, 38, 52): equal lengths would never
trigger drop-out or the refresh exclusion. Comparison is teacher-forced and on
logits, since batching changes RNG draw order for multi-graph batches.

Marked slow: a decode costs ~100s on CPU against ~4s on a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("graphein")

from dynamicmpnn.eval import sampling  # noqa: E402
from dynamicmpnn.eval.inputs import resolve_inputs  # noqa: E402
from dynamicmpnn.eval.pipeline import build_request  # noqa: E402

TARGETS = ["2mwf_2nnt", "6z4u_7kdt", "1bdt_1qtg"]   # N = 25, 38, 52
CHECKPOINT = "checkpoints/multi_chain_reload.ckpt"
NUM_SAMPLES = 1
REFRESH = 4          # must be > 0 so the refresh exclusion is exercised
TOLERANCE = 1e-3

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def batched_fixture(tmp_path_factory):
    from hydra import compose, initialize_config_dir
    from torch_geometric.data import Batch

    from dynamicmpnn import configs

    tmp_path = tmp_path_factory.mktemp("batched_multi_graph")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs, model = [], None

    for target in TARGETS:
        with initialize_config_dir(version_base="1.3",
                                   config_dir=str(Path(configs.__file__).parent)):
            cfg = compose(config_name="evaluate", overrides=[
                f"eval={target}", f"eval.model_ref={CHECKPOINT}",
                f"eval.num_samples={NUM_SAMPLES}", "eval.af3_evaluate=false",
                f"output_dir={tmp_path / target}"])
        request = build_request(cfg)
        bundle = resolve_inputs(request)
        checkpoint = torch.load(str(request.resolved_model_path),
                                map_location="cpu", weights_only=False)
        features_cfg = sampling._checkpoint_config(checkpoint).get("features",
                                                                   cfg.features)
        graphs.append(sampling._create_featuriser(features_cfg)(
            bundle.multi_state_data, pdb_code=target))
        if model is None:
            model = sampling._load_model_for_sampling(
                request.resolved_model_path, cfg, NUM_SAMPLES, checkpoint).to(device)

    singles = [Batch.from_data_list([g]).to(device) for g in graphs]
    combined = Batch.from_data_list(graphs).to(device)
    sizes = [int(b.num_decoder_nodes.sum()) for b in singles]

    generator = torch.Generator().manual_seed(2468)
    forced = [torch.randint(0, model.out_dim, (n,), generator=generator).to(device)
              for n in sizes]
    return model, singles, combined, sizes, forced


def test_targets_have_different_lengths(batched_fixture):
    """Guards the premise: equal lengths would silently stop exercising the
    active-set drop-out and the refresh exclusion."""
    _, _, _, sizes, _ = batched_fixture
    assert len(set(sizes)) == len(sizes), f"targets must differ in length, got {sizes}"


def test_batched_matches_individual_decodes(batched_fixture):
    """The core invariant: each graph's logits must be unaffected by which
    other graphs happen to share its batch."""
    model, singles, combined, sizes, forced = batched_fixture

    with torch.inference_mode():
        individual = []
        for single, force in zip(singles, forced):
            _, logits, _ = model.sample_batched(single, return_logits=True,
                                                refresh_interval=REFRESH,
                                                forced_seq=force)
            individual.append(logits[0])
        _, together, _ = model.sample_batched(combined, return_logits=True,
                                              refresh_interval=REFRESH,
                                              forced_seq=torch.cat(forced))

    assert len(together) == len(TARGETS)
    for target, alone, batched, size in zip(TARGETS, individual, together, sizes):
        assert batched.shape == (NUM_SAMPLES, size, model.out_dim), target
        difference = (alone.float() - batched.float()).abs().max()
        assert difference <= TOLERANCE, (
            f"{target} decoded differently in a batch than alone "
            f"(max_abs={difference:.3e}); batching must not couple graphs.")


def test_sample_reassembles_multi_graph_results(batched_fixture):
    """sample() concatenates per-graph lists into sample_v3's contract. A
    mistake here misaligns residues between graphs without raising."""
    model, _, combined, sizes, _ = batched_fixture
    total = sum(sizes)

    with torch.inference_mode():
        sequences, logits, indices, batch_indices = model.sample(
            combined, return_logits=True)

    assert sequences.shape == (NUM_SAMPLES, total)
    assert logits.shape == (NUM_SAMPLES, total, model.out_dim)
    assert indices.shape[0] == total
    assert batch_indices.shape[0] == total

    # Each graph must claim exactly its own residues, in batch order.
    counts = torch.bincount(batch_indices.cpu(), minlength=len(sizes))
    assert counts.tolist() == sizes, (
        f"batch_indices splits residues as {counts.tolist()}, expected {sizes}")
    assert torch.equal(batch_indices.cpu(), torch.cat([
        torch.full((n,), i, dtype=batch_indices.dtype)
        for i, n in enumerate(sizes)])), "graphs must stay in batch order"

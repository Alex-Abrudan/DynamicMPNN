"""The fast decoders are only exact when decoder layers leave edges untouched.

`GVPConvLayer.forward` rewrites `edge_attr` through `edge_message_func` before
convolving, but only when built with `n_edge_gvps > 0`. `_local_decode` and
`_refresh_decode` skip `GVPConvLayer.forward` and call `layer.conv.message_func`
directly, building edge features once per step and reusing them across layers.

That is exact for `n_edge_gvps == 0` -- how every shipped checkpoint is
configured -- and silently wrong otherwise, because the per-layer edge updates
would never run. So `sample()` must fall back to `sample_v3`, and
`sample_batched` must refuse rather than return quietly wrong numbers.

Builds tiny models directly; no checkpoint, no GPU, no featurisation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dynamicmpnn.modules.models.models import DynamicMPNN  # noqa: E402

COMMON = dict(node_in_dim=(6, 2), node_h_dim=(8, 2), edge_in_dim=(4, 1),
              edge_h_dim=(4, 1), num_encoder_layers=1, num_decoder_layers=1,
              out_dim=20, n_samples=1, drop_rate=0.0)


def _model(n_edge_gvps):
    return DynamicMPNN(n_edge_gvps=n_edge_gvps, **COMMON)


def test_detects_when_decoder_updates_edges():
    assert _model(0)._decoder_updates_edge_features() is False
    assert _model(2)._decoder_updates_edge_features() is True


def test_shipped_configuration_uses_the_fast_path():
    """n_edge_gvps=0 is what every released checkpoint stores, so the fast
    decoder must stay reachable -- a fallback that always fires would quietly
    undo the speedup."""
    model = _model(0)
    assert not model._decoder_updates_edge_features()
    assert any(getattr(layer, "edge_message_func", None) is None
               for layer in model.decoder_layers)


def test_sample_batched_refuses_models_that_update_edges():
    """Direct callers bypass sample()'s fallback, so this must raise."""
    model = _model(2)
    with pytest.raises(NotImplementedError, match="edge-feature"):
        model.sample_batched(batch=None)


def test_edge_message_func_is_present_only_when_configured():
    """Pins the upstream behaviour the fallback depends on: if GVPConvLayer
    ever stopped gating on n_edge_gvps, the fallback would misfire."""
    assert all(getattr(layer, "edge_message_func", None) is not None
               for layer in _model(2).decoder_layers)
    assert all(getattr(layer, "edge_message_func", None) is None
               for layer in _model(0).decoder_layers)

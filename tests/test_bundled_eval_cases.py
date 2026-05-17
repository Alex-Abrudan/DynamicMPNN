from pathlib import Path

from omegaconf import OmegaConf

from dynamicmpnn import constants
from dynamicmpnn.eval.inputs import resolve_inputs
from dynamicmpnn.eval.pipeline import build_request


CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "eval"
PDB_DIR = constants.BENCHMARK_PDB_DIR


def _load_cfg(config_path: Path, model_path: Path, output_dir: Path):
    cfg = OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "eval": OmegaConf.load(config_path),
        }
    )
    cfg.eval.model_ref = str(model_path)
    cfg.eval.af3.executable = "/bin/true"
    return cfg


def test_bundled_eval_cases_cover_expected_set(tmp_path):
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")

    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    pdb_paths = sorted(PDB_DIR.glob("*.pdb"))

    assert len(config_paths) == 96
    assert len(pdb_paths) == 188

    decoy_counts = {"full": 0, "partial": 0, "none": 0}
    partial_decoy_configs: list[str] = []
    no_decoy_configs: list[str] = []

    for config_path in config_paths:
        cfg = _load_cfg(config_path, model_path, tmp_path / config_path.stem)
        request = build_request(cfg)
        bundle = resolve_inputs(request)

        assert request.targets[0].pdb_path.exists()
        assert request.targets[1].pdb_path.exists()

        if request.decoys is None:
            decoy_counts["none"] += 1
            no_decoy_configs.append(config_path.stem)
        else:
            if len(request.decoys) == 2:
                decoy_counts["full"] += 1
            else:
                decoy_counts["partial"] += 1
                partial_decoy_configs.append(config_path.stem)
            for decoy in request.decoys.values():
                assert decoy.pdb_path.exists()

        query_length = bundle.request.alignment.query_length
        assert query_length > 0
        assert len(bundle.request.sequence_recovery_target_sequences) == 2
        for sequence in bundle.request.sequence_recovery_target_sequences:
            assert len(sequence) == query_length

    assert decoy_counts == {"full": 95, "partial": 1, "none": 0}
    assert partial_decoy_configs == ["4cmq_4zt0"]
    assert no_decoy_configs == []

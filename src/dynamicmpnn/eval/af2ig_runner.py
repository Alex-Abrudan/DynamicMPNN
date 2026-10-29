"""AF2 initial-guess predictor, run as a subprocess by eval/af2ig.py.

This module is deliberately standalone: it imports AlphaFold2 and nothing
from dynamicmpnn, because AlphaFold2 needs a jax old enough to conflict with
this package's torch stack and therefore runs in its own environment. It
reads the job manifest that `write_af2ig_manifest` produced and writes, per
job, the predicted structure and its confidences - the same two artifacts the
AF3 path leaves behind, so `score_af2ig_jobs` reads them the same way.

The metric itself is Bennett et al.'s initial guess (dl_binder_design): the
patched `modules.py` does `prev_pos += initial_guess`, so the structure
module starts from a given backbone instead of the black-hole initialisation.
Two details decide whether that bites:

  * `get_prev` overwrites `prev_pos` with the model's own output on every
    recycle, so the seed survives only the first iteration. Fewer recycles
    therefore constrain the prediction more, not less, and the default here
    is one.
  * For a monomer their code templates nothing, which leaves the seed as the
    only structural input and in practice too weak to hold. Passing the same
    backbone to the template stack as well is what makes the metric work.
    The template carries the DESIGNED sequence, never the native one, so no
    residue identity from the reference structure can reach the model - the
    leak that the AF3 template path has by construction.

Set AF2IG_INITIAL_GUESS_DIR to the dl_binder_design af2_initial_guess
checkout that provides the patched alphafold package.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np


def _install_alphafold_path() -> None:
    initial_guess_dir = os.environ.get("AF2IG_INITIAL_GUESS_DIR")
    if initial_guess_dir:
        sys.path.insert(0, str(initial_guess_dir))


def _shim_removed_apis() -> None:
    """Make the bundled AlphaFold importable on a modern jax and Biopython.

    `jax.tree_map` was deprecated in jax 0.4.25 and removed in 0.6, and the
    bundled AlphaFold calls it in about thirty places; the jaxlib that still
    has it needs a cuDNN with no aarch64 wheel. `Bio.Data.SCOPData` went in
    Biopython 1.82 and the mapping it held now lives in Bio.Data.PDBData -
    nothing here parses an mmCIF, but the import still has to resolve because
    alphafold.data.pipeline pulls in templates.
    """
    import jax

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map

    try:
        from Bio.Data import SCOPData  # noqa: F401
    except ImportError:
        from Bio.Data import PDBData

        module = types.ModuleType("Bio.Data.SCOPData")
        module.protein_letters_3to1 = dict(PDBData.protein_letters_3to1_extended)
        sys.modules["Bio.Data.SCOPData"] = module


def empty_template(length: int, residue_constants) -> dict:
    """The monomer case of af2_util.generate_template_features: all zeros."""
    sequence = "-" * length
    return {
        "template_all_atom_positions": np.zeros(
            (1, length, residue_constants.atom_type_num, 3), dtype=np.float32
        ),
        "template_all_atom_masks": np.zeros(
            (1, length, residue_constants.atom_type_num), dtype=np.float32
        ),
        "template_sequence": [sequence.encode()],
        "template_aatype": np.array(
            residue_constants.sequence_to_onehot(sequence, residue_constants.HHBLITS_AA_TO_ID)
        )[None],
        "template_confidence_scores": np.full((1, length), -1, dtype=np.int64),
        "template_domain_names": ["none".encode()],
        "template_release_date": ["none".encode()],
    }


def backbone_template(sequence: str, positions: np.ndarray, masks: np.ndarray, residue_constants) -> dict:
    """generate_template_features with every residue templated on the seed.

    The sequence written into the template is the designed one, so the
    template carries geometry and nothing else.
    """
    return {
        "template_all_atom_positions": positions[None].astype(np.float32),
        "template_all_atom_masks": masks[None].astype(np.float32),
        "template_sequence": [sequence.encode()],
        "template_aatype": np.array(
            residue_constants.sequence_to_onehot(sequence, residue_constants.HHBLITS_AA_TO_ID)
        )[None],
        "template_confidence_scores": np.full((1, len(sequence)), 9, dtype=np.int64),
        "template_domain_names": ["none".encode()],
        "template_release_date": ["none".encode()],
    }


def seed_from_guess(job: dict, residue_constants) -> tuple[np.ndarray, np.ndarray]:
    """Scatter a stored backbone into atom37 slots of the designed sequence.

    Mirrors what AF3 does with queryIndices/templateIndices: residue
    `template_indices[i]` of the stored structure lands at position
    `query_indices[i]` of the sequence being folded, and everything else
    stays masked out.
    """
    data = np.load(job["guess_npz"])
    xyz, mask = data["xyz"], data["mask"]
    atoms = [name.decode() if isinstance(name, bytes) else str(name) for name in data["atoms"]]

    query_indices = np.asarray(job["query_indices"], dtype=np.int64)
    template_indices = np.asarray(job["template_indices"], dtype=np.int64)
    if query_indices.size != template_indices.size:
        raise ValueError(f"{job['job_name']}: query and template index counts differ")

    length = len(job["sequence"])
    if query_indices.size and int(query_indices.max()) >= length:
        raise ValueError(
            f"{job['job_name']}: query mapping needs index {int(query_indices.max())} "
            f"of a {length}-residue sequence"
        )
    if template_indices.size and int(template_indices.max()) >= xyz.shape[0]:
        raise ValueError(
            f"{job['job_name']}: template mapping needs residue {int(template_indices.max())} "
            f"of a {xyz.shape[0]}-residue structure"
        )

    positions = np.zeros((length, residue_constants.atom_type_num, 3), dtype=np.float32)
    masks = np.zeros((length, residue_constants.atom_type_num), dtype=np.float32)
    for atom_index, atom_name in enumerate(atoms):
        slot = residue_constants.atom_order[atom_name]
        selected_mask = mask[template_indices, atom_index]
        positions[query_indices, slot] = xyz[template_indices, atom_index] * selected_mask[:, None]
        masks[query_indices, slot] = selected_mask
    return positions, masks


def write_prediction(output_dir: Path, sequence: str, result: dict, chain_id: str, residue_constants, protein) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atom_positions = np.asarray(result["structure_module"]["final_atom_positions"])
    atom_mask = np.asarray(result["structure_module"]["final_atom_mask"])
    plddt = np.asarray(result["plddt"], dtype=np.float64)
    aatype = np.array(
        [residue_constants.restype_order_with_x.get(residue, residue_constants.restype_num) for residue in sequence],
        dtype=np.int32,
    )
    structure = protein.Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(1, len(sequence) + 1, dtype=np.int32),
        b_factors=np.repeat(plddt[:, None], atom_positions.shape[1], axis=1),
    )
    pdb_text = protein.to_pdb(structure)
    if chain_id != "A":
        # alphafold.common.protein always emits chain A for a monomer.
        pdb_text = "\n".join(
            line[:21] + chain_id + line[22:] if line.startswith(("ATOM", "HETATM", "TER")) else line
            for line in pdb_text.splitlines()
        )
    (output_dir / "prediction.pdb").write_text(pdb_text, encoding="utf-8")

    confidences = {
        "plddt": [float(value) for value in plddt],
        "mean_plddt": float(plddt.mean()),
    }
    if "ptm" in result:
        confidences["ptm"] = float(result["ptm"])
    if "predicted_aligned_error" in result:
        confidences["mean_pae"] = float(np.mean(result["predicted_aligned_error"]))
    (output_dir / "confidences.json").write_text(json.dumps(confidences), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--params_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    args = parser.parse_args()

    _install_alphafold_path()
    _shim_removed_apis()

    from alphafold.common import protein, residue_constants
    from alphafold.data import pipeline
    from alphafold.model import config, data, model

    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))["jobs"]
    if not jobs:
        return 0

    recycles = {job["num_recycles"] for job in jobs}
    if len(recycles) != 1:
        raise ValueError(f"One batch must share a recycle count, got {sorted(recycles)}")
    num_recycles = recycles.pop()

    model_config = config.model_config(args.model_name)
    model_config.data.eval.num_ensemble = 1
    model_config.data.common.num_recycle = num_recycles
    model_config.model.num_recycle = num_recycles
    # Single-sequence mode: the designed sequence is the entire MSA.
    model_config.data.common.max_extra_msa = 5
    model_config.data.eval.max_msa_clusters = 5
    model_config.model.embeddings_and_evoformer.initial_guess = True
    model_params = data.get_model_haiku_params(model_name=args.model_name, data_dir=args.params_dir)
    runner = model.RunModel(model_config, model_params)

    base_features: dict[str, dict] = {}
    failures = 0
    for index, job in enumerate(jobs, start=1):
        output_dir = Path(job["output_dir"])
        try:
            sequence = job["sequence"]
            length = len(sequence)
            if sequence not in base_features:
                base_features[sequence] = {
                    **pipeline.make_sequence_features(sequence=sequence, description="none", num_res=length),
                    **pipeline.make_msa_features(msas=[[sequence]], deletion_matrices=[[[0] * length]]),
                }
            base = base_features[sequence]
            positions, masks = seed_from_guess(job, residue_constants)

            template = (
                backbone_template(sequence, positions, masks, residue_constants)
                if job["template_backbone"]
                else empty_template(length, residue_constants)
            )
            processed = runner.process_features({**base, **template}, random_seed=0)
            result = runner.predict(processed, initial_guess=positions)
            write_prediction(output_dir, sequence, result, job["chain_id"], residue_constants, protein)
        except Exception as exc:  # noqa: BLE001 - one bad job must not sink the batch
            failures += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            print(f"[{index}/{len(jobs)}] {job['job_name']} FAILED: {exc}", flush=True)
            continue
        if index % 25 == 0 or index == len(jobs):
            print(f"[{index}/{len(jobs)}] {job['job_name']}", flush=True)

    print(f"{len(jobs) - failures}/{len(jobs)} predictions written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

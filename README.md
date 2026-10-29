# DynamicMPNN

DynamicMPNN is a deep learning inverse folding model that generates protein sequences compatible with multiple conformational states. Built on GVP-GNN architecture, it pools structural information across conformations during encoding and uses autoregressive decoding to predict amino acid sequences. We introduce a multi-state self-consistency metric using template-based AlphaFold3 structure prediction with decoy normalization to evaluate design success.

![DynamicMPNN pipeline](docs/pipeline.png)

## Installation

```bash
conda env create -f environment.yml
conda activate dynamicmpnn
pip install -e .
```

## Configuration

Create a `.env` file from the template and update paths for your machine:

```bash
cp .env.example .env
```

Required paths:
- `PROJECT_PATH` - Path to this repository
- `PUBLIC_DB` - Directory containing processed `.pt` files (e.g., `data/`)
- `RUNS_PATH` - Where training runs/checkpoints are saved

Optional (for AF3 evaluation):
- `AF3_EXECUTABLE`, `AF3_SCRIPT`, `AF3_MODEL_DIR`, `AF3_DB_DIR`

Optional (for AF2 initial-guess evaluation):
- `AF2IG_PYTHON`, `AF2IG_PARAMS_DIR`, `AF2IG_INITIAL_GUESS_DIR`

## Data

Download processed `.pt` files from [Zenodo](https://zenodo.org/records/19687631) and extract to `data/`:

```bash
# Multi-chain
data/train_pt_multi_chain/
data/val_pt_multi_chain/
data/test_pt_multi_chain/

# Single-chain
data/train_pt_single_chain/
data/val_pt_single_chain/
data/test_pt_single_chain/
```

Update `PUBLIC_DB` in `.env` to point to `data/`.

## Checkpoints

Pre-trained model checkpoints are available in the `checkpoints/` directory. Each checkpoint contains model weights, hyperparameters, and optimizer state.

| Checkpoint | Description |
|------------|-------------|
| `multi_chain_reload.ckpt` | Multi-chain mode, 2-state with full PDB encoding, per-epoch cluster resampling |
| `multi_chain_no_reload.ckpt` | Multi-chain mode, 2-state with full PDB encoding, static cluster sampling |
| `single_chain_k*.ckpt` | Single-chain mode, supports up to 5 conformational states (k=2,3,5 available) |

## Training

### Setup

1. Download processed `.pt` files from [Zenodo](https://zenodo.org/records/19687631) to `data/`
2. Configure paths in `.env`

### Experiments

| Experiment | Description |
|------------|-------------|
| `seq30_reload` | Multi-chain, per-epoch cluster resampling |
| `seq30_no_reload` | Multi-chain, static cluster sampling |
| `single_chain_k_conf` | Single-chain, k conformations pooling |

### Commands

```bash
cd src/dynamicmpnn

# Full training
python train.py experiment=seq30_reload

# Quick test run
python train.py experiment=seq30_reload trainer.max_epochs=1 trainer.limit_train_batches=10

# Resume from checkpoint
python train.py experiment=seq30_reload ckpt_path=path/to/checkpoint.ckpt
```

## Evaluation

### Sequence Sampling

Generate sequences for a multi-state target:

```bash
dynamicmpnn-evaluate \
  eval=1bdt_1qtg \
  eval.model_ref=checkpoints/multi_chain_reload.ckpt \
  output_dir=runs/1bdt_1qtg
```

Key parameters:
- `eval.num_samples`: Number of sequences to generate (default: 25)
- `eval.temperature`: Sampling temperature

Output: `samples/samples.csv` and `samples/samples.fasta`

See example configs in `src/dynamicmpnn/configs/eval/` for custom targets.

### AF3 Self-Consistency Metrics

Enable AlphaFold3 structure prediction to compute self-consistency metrics:

```bash
dynamicmpnn-evaluate \
  eval=1bdt_1qtg \
  eval.model_ref=checkpoints/multi_chain_reload.ckpt \
  eval.af3_evaluate=true \
  output_dir=runs/1bdt_1qtg
```

Requires AF3 paths configured in `.env`.

Metrics computed:
- **TM-score**: Template modeling score (Kabsch-aligned)
- **LDDT**: Local distance difference test
- **RMSD**: Root mean square deviation
- **pLDDT**: Predicted local distance difference test

Decoy normalization is used for fair comparison. Output: `summary_af3_results.csv`

### AF2 Initial-Guess Metrics

An alternative refoldability metric that cannot see the native sequence. The
AF3 template path hands AlphaFold3 a template mmCIF carrying the *native*
residue identities; this path seeds AlphaFold2's recycling positions with the
backbone instead (Bennett et al., `dl_binder_design`) and templates the same
backbone under the **designed** sequence, so only geometry reaches the model.

```bash
dynamicmpnn-evaluate \
  eval=1bdt_1qtg \
  eval.model_ref=checkpoints/multi_chain_reload.ckpt \
  eval.af2ig_evaluate=true \
  output_dir=runs/1bdt_1qtg
```

It runs alongside `eval.af3_evaluate` or on its own, produces the same four
metrics on the same residues, and uses the configured decoys the same way: a
decoy job seeds on the decoy backbone and is scored against the decoy's own
coordinates, so `*_decoy_normalized_*` means exactly what it means for AF3.
Output: `summary_af2ig_results.csv`.

AlphaFold2 needs a jax that conflicts with this package's torch stack, so the
predictor runs as a subprocess under `eval.af2ig.executable` — the same
arrangement `eval.af3.executable` uses. Two settings matter:

- `eval.af2ig.num_recycles` (default 1). The seed survives only the first
  recycle, because `get_prev` overwrites `prev_pos` with the model's own
  output, so *fewer* recycles constrain the prediction more.
- `eval.af2ig.template_backbone` (default true). For a monomer the seed alone
  is too weak to hold; passing the same backbone to the template stack is what
  makes the metric bite.

## Data Preprocessing (optional)

To regenerate `.pt` files from scratch, start from the AlphaFold mmCIF database. See [`src/dynamicmpnn/scripts/README.md`](src/dynamicmpnn/scripts/README.md) for the full pipeline.

## Citation

```bibtex
@inproceedings{abrudan_dynamicmpnn,
  title={Multi-state Protein Sequence Design with DynamicMPNN},
  author={Abrudan, Alex and Ojeda, Sebastian Pujalte and Joshi, Chaitanya K and Greenig, Matthew and Engelberger, Felipe and Khmelinskaia, Alena and Meiler, Jens and Vendruscolo, Michele and Knowles, Tuomas},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

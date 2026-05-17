# DynamicProt Data Preprocessing Scripts

Scripts for building the training data from raw PDB structures.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1-2: Sequence Clustering                          [clustering/]     │
│  bash cluster.sh <input> <threshold> <mode>                                 │
│    → mmseqs2 clustering at 30/40/80/90% identity                            │
│    → then: python build_cluster_csv.py → seqres_simple_clusters.csv         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Gather Sequences                               [pipeline/]       │
│  sbatch slurm_gather_seq80                                                  │
│    → combines CIFs per cluster80                                            │
│    → ClustalOmega alignment                                                 │
│    → output: pdb_cache_seq80/{cluster80}/                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Process to .pt                                 [pipeline/]       │
│  sbatch slurm_process_pt_seq80                                              │
│    → featurizes each cluster                                                │
│    → output: train_pt_multi_chain/{cluster80}.pt                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Foldseek TM-scores                             [foldseek/]       │
│  sbatch slurm_tm_foldseek.sh (create your own wrapper)                      │
│    → all-vs-all structural alignment per cluster                            │
│    → output: foldseek_alignments/{cluster80}/                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: Enrich .pt with TM-scores                      [pipeline/]       │
│  sbatch slurm_add_TM_scores_seq80                                           │
│    → reads foldseek output, adds tm_scores to .pt                           │
│    → output: train_pt_multi_chain/{cluster80}.pt                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: Train/Val/Test Split                           [clustering/]     │
│  python build_train_val_test_split.py                                       │
│    → excludes val/test cluster30s from training                             │
│    → output: data/train_seq80_pool_filtered.csv                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 8: Val/Test .pt Files                             [splits/]         │
│  python build_val_test_pts.py                                               │
│    → creates pseudo-cluster .pts from CoDNaS pairs                          │
│    → val: 900001-900100, test: 800001-800100                                │
│    → output: val_pt_multi_chain/, test_pt_multi_chain/          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  OPTIONAL: Curriculum Precompute                         [splits/]         │
│  sbatch slurm_precompute_min_tm.sh                                          │
│    → adds min_tm column for curriculum learning                             │
│    → output: data/train_seq80_pool_filtered_mintm.csv                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
scripts/
├── clustering/     # Stages 1-2, 7: mmseqs2 clustering + train/val/test split
├── pipeline/       # Stages 3-4, 6: gather → process → enrich with TM
├── foldseek/       # Stage 5: structural alignment & TM-score computation
├── splits/         # Stage 8 + optional: val/test .pts, curriculum
├── validation/     # Validation utilities and sample repair
├── download/       # Data download scripts
└── legacy/         # Old/unused scripts (kept for reference)
```

### clustering/

| Script | Purpose |
|--------|---------|
| `cluster.sh` | Runs mmseqs2 easy-cluster for one threshold (usage: `bash cluster.sh <input> <threshold> <mode>`) |
| `build_cluster_csv.py` | Joins cluster TSVs → `seqres_simple_clusters.csv` |
| `build_train_val_test_split.py` | Removes val/test cluster30s → `train_seq80_pool_filtered.csv` |

Environment: `CLUSTERING_BASE` (set in `.env`, required)

### pipeline/

| Script | Purpose |
|--------|---------|
| `gather_sequences_seq80.py` | Combines CIFs per cluster80 + ClustalOmega alignment |
| `process_pt_seq80.py` | Featurizes clusters → `.pt` files |
| `add_TM_scores_seq80.py` | Enriches `.pt` with TM-scores from foldseek |
| `slurm_gather_seq80` | SLURM array wrapper for gather |
| `slurm_process_pt_seq80` | SLURM array wrapper for process |
| `slurm_add_TM_scores_seq80` | SLURM array wrapper for TM enrichment |

Single-chain variants: `*_single.py` and `*_single.sh`

### foldseek/

| Script | Purpose |
|--------|---------|
| `tm_foldseek.py` | Runs foldseek all-vs-all per cluster (case-sensitive chain matching) |
| `tm_parse_foldseek.py` | Parses foldseek TSV output |
| `constants.py` | Shared paths and configuration |

Tracks processed clusters in `processed_clusters.txt` to allow safe re-runs.

### splits/

| Script | Purpose |
|--------|---------|
| `build_val_test_pts.py` | Creates val/test `.pt` from CoDNaS pairs (IDs 900001-900100, 800001-800100) |
| `precompute_min_tm.py` | Adds `min_tm` column for curriculum learning |
| `slurm_precompute_min_tm.sh` | SLURM wrapper for min_tm precompute |

Single-chain variant: `build_val_test_pts_single.py`

### validation/

| Script | Purpose |
|--------|---------|
| `validate_clus80_samples.py` | Finds bad samples → `failed_clus80.txt` |
| `validate_single_chain_samples.py` | Single-chain validation |
| `validate_training_samples.py` | Validates training samples load correctly |
| `remove_cluster_members.py` | Surgically removes bad members from `.pt` files |
| `test_model_integration.py` | Integration tests for model loading |

### download/

| Script | Purpose |
|--------|---------|
| `download_codnas_mmcif.py` | Downloads CoDNaS mmCIF files |

### legacy/

Old scripts kept for reference. These use hardcoded paths to old locations and are not part of the current pipeline.

## Multi-chain vs Single-chain

Most scripts have `*_single.py` variants for single-chain data:
- Uses `train_pt_multi_chain_single/` and `train_pt_single_chain/`
- Val/test IDs are the same (900001-900100, 800001-800100)
- Training CSV: `train_single_chain_pool.csv`

## Quick Start

```bash
# Full pipeline (after raw mmCIFs are ready):
cd src/dynamicmpnn/scripts

# 1-2. Cluster sequences (run for each threshold: 0.3, 0.4, 0.8, 0.9)
for thresh in 0.3 0.4 0.8 0.9; do
    bash clustering/cluster.sh seqres $thresh simple
done
python clustering/build_cluster_csv.py seqres simple

# 3. Gather sequences per cluster
python pipeline/gather_sequences_seq80.py

# 4. Process to .pt
python pipeline/process_pt_seq80.py

# 5. Run foldseek for TM-scores
python foldseek/tm_foldseek.py

# 6. Add TM scores to .pt files
python pipeline/add_TM_scores_seq80.py

# 7. Build train split
python clustering/build_train_val_test_split.py

# 8. Build val/test .pts
python splits/build_val_test_pts.py
```

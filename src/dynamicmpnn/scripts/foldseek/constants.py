import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Use environment variables with defaults for external paths
MMCIF_DIR = Path(os.environ.get("MMCIF_DIR", "data/mmcif_files"))
EXAMPLE_CIF = MMCIF_DIR / "4ye4.cif"

ATOMWORKS_METADATA_DIR = DATA_DIR / "atomworks_metadata"
PN_UNITS_FILE = ATOMWORKS_METADATA_DIR / "pn_units_df.parquet"
INTERFACE_FILE = ATOMWORKS_METADATA_DIR / "interfaces_df.parquet"
CHAINS_DIR = DATA_DIR / "rcsb_chains"

DPROT_DIR = Path(os.environ.get("DPROT_DIR", "."))
SINGLE_FEAT_DIR = Path(os.environ.get("SINGLE_FEAT_DIR", "data/single"))

CODNAS_MMCIF_DIR = Path(os.environ.get("CODNAS_MMCIF_DIR", "data/codnas/mmcif_files"))
CODNAS_TEST_SET = DATA_DIR / "codnas" / "test_set.csv"
CODNAS_VAL_SET = DATA_DIR / "codnas" / "val_set.csv"
CODNAS_ALL = DATA_DIR / "codnas" / "codnas-2025.csv"

USALIGN_DIR = Path(os.environ.get("USALIGN_DIR", "bin/USalign"))
DATA_CLUS_DIR = Path(os.environ.get("DATA_CLUS_DIR", "data/pdb_align"))

# Foldseek paths
FOLDSEEK_BIN = Path(os.environ.get("FOLDSEEK_BIN", "foldseek"))
FOLDSEEK_CLUS_DIR = Path(os.environ.get("FOLDSEEK_CLUS_DIR", "data/foldseek_align"))
REDUCED_MMCIF_DIR = Path(os.environ.get("REDUCED_MMCIF_DIR", "data/reduced_mmcifs"))

# mmseqs2 seqres 80% clusters
MMSEQS2_CLUSTERS_CSV = Path(os.environ.get("MMSEQS2_CLUSTERS_CSV", "data/seqres_simple_clusters.csv"))
FOLDSEEK_CLUS_DIR_MMSEQS2 = Path(os.environ.get("FOLDSEEK_CLUS_DIR_MMSEQS2", "data/foldseek_alignments"))

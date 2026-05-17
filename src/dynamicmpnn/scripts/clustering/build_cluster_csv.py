#!/usr/bin/env python
"""Build a CSV joining entity/chain mapping, SEQRES sequences, and cluster IDs.

Usage: python build_cluster_csv.py [seqres|observed] [simple|sensitive]
       python build_cluster_csv.py                    # runs all combinations

Env:   CLUSTERING_BASE - base directory for clustering data
"""

import csv
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("CLUSTERING_BASE", "data/clustering"))
EXTRACT_DIR = BASE / "seq_extraction/output"
THRESHOLDS = [30, 40, 80, 90]


def load_clusters(tsv_path):
    """Load cluster TSV → dict mapping member entity to integer cluster ID."""
    rep_to_id = {}
    member_to_id = {}
    next_id = 0
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep not in rep_to_id:
                rep_to_id[rep] = next_id
                next_id += 1
            member_to_id[member] = rep_to_id[rep]
    return member_to_id


def load_sequences(fasta_path):
    """Load FASTA → dict mapping entity key to sequence."""
    seqs = {}
    key = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
                seqs[key] = []
            else:
                seqs[key].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def build_csv(seq_type, sensitivity, seqs):
    cluster_dir = BASE / "seq_clustering/output" / seq_type / sensitivity
    out_dir = BASE / "seq_clustering/output/collated"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"{seq_type}_{sensitivity}_clusters.csv"

    print(f"\n[{seq_type} / {sensitivity}]")

    clusters = {}
    for t in THRESHOLDS:
        tsv = cluster_dir / f"id{t}" / f"{seq_type}_cluster.tsv"
        clusters[t] = load_clusters(tsv)
        print(f"  id{t}: {len(set(clusters[t].values())):,} clusters")

    written = skipped = 0
    with open(EXTRACT_DIR / "entity_chain_map.csv") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["pdb_chain", "pdb_auth", "pdb_entity"]
                        + [f"cluster{t}" for t in THRESHOLDS]
                        + ["exptl_method", "x_frac", "sequence"])

        for row in reader:
            pdb_chain = f"{row['pdb_id']}_{row['chain_id']}"
            entity_key = f"{row['pdb_id']}_{row['entity_id']}"
            # observed uses chain IDs, seqres uses entity IDs
            lookup_key = pdb_chain if seq_type == "observed" else entity_key
            seq = seqs.get(lookup_key, "")
            cids = [clusters[t].get(lookup_key) for t in THRESHOLDS]
            if not seq or any(c is None for c in cids):
                skipped += 1
                continue
            writer.writerow([f"{row['pdb_id']}_{row['chain_id']}",
                             f"{row['pdb_id']}_{row['author_chain_id']}",
                             entity_key] + cids
                            + [row.get("exptl_method", ""), row.get("x_frac", ""), seq])
            written += 1

    print(f"  → {written:,} rows, {skipped:,} skipped → {out_csv.name}")


def main():
    if len(sys.argv) == 3:
        seq_types = [sys.argv[1]]
        sensitivities = [sys.argv[2]]
    else:
        seq_types = ["seqres", "observed"]
        sensitivities = ["simple", "sensitive"]

    for seq_type in seq_types:
        print(f"Loading {seq_type} sequences...")
        seqs = load_sequences(EXTRACT_DIR / f"{seq_type}.fasta")
        for sensitivity in sensitivities:
            build_csv(seq_type, sensitivity, seqs)


if __name__ == "__main__":
    main()

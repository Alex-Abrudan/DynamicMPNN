"""The vectorised Needleman-Wunsch must match the scalar recurrence exactly.

`_sequence_similarity` feeds `similarity <= SEQ_SIMILARITY_THRESHOLD`, which
decides whether a context chain's residues are sequence-masked. A near-threshold
difference therefore changes the graph the model sees, silently. "Close enough"
is not good enough here, so these tests demand exact equality against the
original scalar implementation, kept verbatim below as the oracle.

Two things can go wrong independently:

* the DP itself -- the same-row `score[i, j-1]` dependency was replaced by a
  running-maximum identity, which is where an off-by-one would hide;
* the traceback -- it reads the score matrix and breaks ties by a fixed
  priority, so an altered matrix can pick a different equal-scoring alignment
  with a different match count. Comparing the returned percentage (not the
  matrix) exercises both.

Tie-heavy inputs get their own cases because that is where traceback choices
actually differ. Pure Python oracle, so lengths are kept small; the suite runs
in about a second.
"""

from __future__ import annotations

import random

import pytest

numpy = pytest.importorskip("numpy")

from dynamicmpnn.eval.inputs import _sequence_similarity  # noqa: E402

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def scalar_reference(seq1: str, seq2: str) -> float:
    """The original implementation, kept verbatim as the oracle."""
    match_score, mismatch_score, gap_penalty = 1, -1, -1
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = numpy.zeros((rows, cols), dtype=numpy.int32)
    score_matrix[:, 0] = numpy.arange(rows) * gap_penalty
    score_matrix[0, :] = numpy.arange(cols) * gap_penalty

    for i in range(1, rows):
        for j in range(1, cols):
            match = score_matrix[i - 1, j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
            delete = score_matrix[i - 1, j] + gap_penalty
            insert = score_matrix[i, j - 1] + gap_penalty
            score_matrix[i, j] = max(match, delete, insert)

    aligned1: list[str] = []
    aligned2: list[str] = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i - 1, j - 1]
                + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)):
            aligned1.append(seq1[i - 1]); aligned2.append(seq2[j - 1]); i -= 1; j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + gap_penalty:
            aligned1.append(seq1[i - 1]); aligned2.append("-"); i -= 1
        else:
            aligned1.append("-"); aligned2.append(seq2[j - 1]); j -= 1

    aligned1.reverse(); aligned2.reverse()
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b)
    return (matches / len(aligned1)) * 100 if aligned1 else 0.0


@pytest.mark.parametrize("seq1,seq2", [
    ("", ""),                       # both empty
    ("A", ""), ("", "A"),           # one empty: pure-gap alignment
    ("A", "A"), ("A", "C"),         # single match / single mismatch
    ("ACDEF", "ACDEF"),             # identical
    ("ACDEF", "FEDCA"),             # reversed: no useful alignment
    ("AAAA", "AA"),                 # length mismatch, needs gaps
    ("ACDEF", "ACXDEF"),            # single insertion
    ("ACXDEF", "ACDEF"),            # single deletion
])
def test_matches_scalar_reference_on_edge_cases(seq1, seq2):
    assert _sequence_similarity(seq1, seq2) == scalar_reference(seq1, seq2)


@pytest.mark.parametrize("seq1,seq2", [
    ("AC", "ACXXXXXXXX"),           # long run of insertions
    ("ACXXXXXXXX", "AC"),           # long run of deletions
    ("A", "AAAAAAAAAAAA"),          # extreme length asymmetry
    ("ACDEF", "ACDEFXXXXXXXXXX"),   # trailing insertion run
    ("XXXXXXXXXXACDEF", "ACDEF"),   # leading deletion run
    ("ACDEFACDEF", "AC"),           # one short, many gaps needed
])
def test_matches_scalar_reference_on_consecutive_gap_runs(seq1, seq2):
    """The same-row `score[i, j-1]` dependency was replaced by a running
    maximum, and that identity only matters when SEVERAL consecutive gaps in
    one row are optimal. Single-insertion cases barely exercise it: mutation
    testing showed that dropping the running maximum altogether was caught by
    only one of the original cases. These are the inputs that make it bite.
    """
    assert _sequence_similarity(seq1, seq2) == scalar_reference(seq1, seq2)


@pytest.mark.parametrize("seq1,seq2", [
    ("AAAAAAAA", "AAAAAAAA"),       # every cell ties
    ("AAAAAAAA", "AAAA"),           # gap placement is arbitrary
    ("ABABABAB", "BABABABA"),       # alternating: many equal-scoring paths
    ("AAAACCCC", "CCCCAAAA"),       # two equally good blocks
    ("AAAA", "CCCC"),               # all mismatch, gaps and mismatches tie
    ("AACAA", "AAGAA"),             # single central mismatch, symmetric
])
def test_matches_scalar_reference_on_tie_heavy_inputs(seq1, seq2):
    """Ties are where the traceback can legitimately pick different alignments
    of equal score, so an altered score matrix shows up here first."""
    assert _sequence_similarity(seq1, seq2) == scalar_reference(seq1, seq2)


def test_matches_scalar_reference_on_random_pairs():
    """Randomised sweep across relatedness levels, including near-identical
    pairs that land close to SEQ_SIMILARITY_THRESHOLD."""
    rng = random.Random(20261026)
    mismatches = []
    for _ in range(40):
        length_a, length_b = rng.randint(1, 60), rng.randint(1, 60)
        relatedness = rng.choice([0.0, 0.3, 0.6, 0.85, 0.95, 1.0])
        base = "".join(rng.choice(AMINO_ACIDS) for _ in range(length_a))
        if relatedness <= 0:
            other = "".join(rng.choice(AMINO_ACIDS) for _ in range(length_b))
        else:
            chars = list(base[:length_b].ljust(length_b, "A"))
            for index in range(len(chars)):
                if rng.random() > relatedness:
                    chars[index] = rng.choice(AMINO_ACIDS)
            other = "".join(chars)
        actual, expected = _sequence_similarity(base, other), scalar_reference(base, other)
        if actual != expected:
            mismatches.append((base, other, expected, actual))
    assert not mismatches, f"{len(mismatches)} pair(s) diverged: {mismatches[:3]}"


def test_asymmetric_lengths_both_orders():
    """The running-maximum runs along rows; swapping the arguments transposes
    the problem and would expose an axis mix-up."""
    rng = random.Random(7)
    for _ in range(10):
        seq1 = "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(1, 40)))
        seq2 = "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(1, 40)))
        assert _sequence_similarity(seq1, seq2) == scalar_reference(seq1, seq2)
        assert _sequence_similarity(seq2, seq1) == scalar_reference(seq2, seq1)

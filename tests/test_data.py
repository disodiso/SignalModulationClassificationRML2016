"""Sanity checks on the dataset plumbing, run against a synthetic stand-in.

The real dataset is a 600 MB download, so these tests build a dict with the
same structure and exercise the shape, ordering and stratification guarantees
the rest of the project relies on.
"""

import numpy as np
import pytest

from src.data import Split, split, stratification_key, to_arrays

MODS = ["BPSK", "QPSK", "QAM16"]
SNRS = [-20, -10, 0, 10, 18]
PER_CELL = 40


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    # Insertion order deliberately shuffled: to_arrays must not depend on it.
    keys = [(mod, snr) for mod in MODS for snr in SNRS]
    rng.shuffle(keys)
    return {key: rng.standard_normal((PER_CELL, 2, 128)).astype(np.float32) for key in keys}


def test_to_arrays_shapes_and_labels(raw):
    X, y, snr, mods = to_arrays(raw)

    assert X.shape == (len(MODS) * len(SNRS) * PER_CELL, 2, 128)
    assert X.dtype == np.float32
    assert mods == sorted(MODS)
    assert set(np.unique(y)) == set(range(len(MODS)))
    assert set(np.unique(snr)) == set(SNRS)
    # Every class appears equally often, one label id per modulation.
    assert np.bincount(y).tolist() == [len(SNRS) * PER_CELL] * len(MODS)


def test_to_arrays_is_insertion_order_independent(raw):
    reordered = dict(reversed(list(raw.items())))
    for a, b in zip(to_arrays(raw), to_arrays(reordered), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_stratification_key_is_injective_over_pairs():
    y = np.array([0, 0, 1, 1, 2])
    snr = np.array([-20, 18, -20, 18, -20])
    key = stratification_key(y, snr)
    assert len(np.unique(key)) == 5


def test_split_partitions_without_overlap(raw):
    X, y, snr, _ = to_arrays(raw)
    s = split(X, y, snr, val_size=0.2, test_size=0.2, seed=1)

    assert isinstance(s, Split)
    sizes = s.sizes()
    assert sum(sizes.values()) == len(y)
    assert sizes["val"] == sizes["test"] == pytest.approx(0.2 * len(y), abs=1)

    # No example may appear in more than one partition.
    flat = [part.reshape(len(part), -1) for part in (s.X_train, s.X_val, s.X_test)]
    rows = {tuple(row) for part in flat for row in part}
    assert len(rows) == len(y)


def test_split_preserves_class_and_snr_balance(raw):
    X, y, snr, _ = to_arrays(raw)
    s = split(X, y, snr, val_size=0.2, test_size=0.2, seed=1)

    for labels, snrs in ((s.y_train, s.snr_train), (s.y_val, s.snr_val), (s.y_test, s.snr_test)):
        counts = np.bincount(stratification_key(labels, snrs))
        # Stratified splitting keeps every (class, SNR) cell within rounding.
        assert counts.max() - counts.min() <= 1
        assert len(counts) == len(MODS) * len(SNRS)


def test_split_without_validation_matches_paper_protocol(raw):
    X, y, snr, _ = to_arrays(raw)
    s = split(X, y, snr, val_size=0.0, test_size=0.5)

    assert s.sizes() == {"train": len(y) // 2, "val": 0, "test": len(y) // 2}
    assert s.X_val.shape[1:] == X.shape[1:]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_size": -0.1},
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"val_size": 0.6, "test_size": 0.5},
    ],
)
def test_split_rejects_impossible_fractions(raw, kwargs):
    X, y, snr, _ = to_arrays(raw)
    with pytest.raises(ValueError):
        split(X, y, snr, **kwargs)

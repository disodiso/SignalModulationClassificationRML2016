"""Loading and splitting for the RadioML2016.10a dataset.

Expected file: data/RML2016.10a_dict.pkl
Format: pickle dict keyed by (modulation_type: str, snr: int) -> ndarray
of shape (num_examples, 2, 128) holding I/Q samples.
"""

import pickle

import numpy as np
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_raw(path: str = "data/RML2016.10a_dict.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def to_arrays(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten the (mod, snr) -> examples dict into parallel arrays.

    Returns X (N, 2, 128), y (N,) integer label ids, snr (N,) ints,
    and the list of modulation class names indexed by label id.
    """
    mods = sorted({mod for mod, _snr in raw.keys()})
    mod_to_id = {mod: i for i, mod in enumerate(mods)}

    X_parts, y_parts, snr_parts = [], [], []
    for (mod, snr), examples in raw.items():
        X_parts.append(examples)
        y_parts.append(np.full(len(examples), mod_to_id[mod]))
        snr_parts.append(np.full(len(examples), snr))

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    snr = np.concatenate(snr_parts).astype(np.int64)
    return X, y, snr, mods


def split(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Stratified split on (label, snr) so every class/SNR pair is
    represented proportionally in both train and test sets — this is
    the standard evaluation protocol for RML2016 (accuracy vs. SNR).
    """
    strata = y * 1000 + snr  # combine into one stratification key
    return train_test_split(
        X, y, snr, test_size=test_size, random_state=seed, stratify=strata
    )


if __name__ == "__main__":
    set_seed()
    raw = load_raw()
    X, y, snr, mods = to_arrays(raw)
    print(f"Loaded {X.shape[0]} examples, {len(mods)} classes: {mods}")
    print(f"SNR range: {snr.min()} to {snr.max()} dB")
    X_train, X_test, y_train, y_test, snr_train, snr_test = split(X, y, snr)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

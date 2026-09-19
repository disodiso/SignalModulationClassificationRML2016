"""Loading and splitting for the RadioML 2016.10a dataset.

The dataset is fetched with kagglehub (cached under ``~/.cache/kagglehub``
after the first download) rather than stored in this repository. On disk it is
a pickle holding a dict keyed by ``(modulation: str, snr: int)`` and valued by
an ndarray of shape ``(num_examples, 2, 128)`` of I/Q samples: 11 modulations
x 20 SNRs x 1000 examples.

Note that the pickle is loaded with ``latin1`` encoding because it was written
by Python 2, and that unpickling executes arbitrary code -- only ever point
this module at the official dataset or a file you produced yourself.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "nolasthitnotomorrow/radioml2016-deepsigcom"
DATASET_FILENAME = "RML2016.10a_dict.pkl"


@dataclass(frozen=True)
class Split:
    """One train/validation/test partition, with SNR kept alongside labels.

    The SNR of each example is not a network input; it is carried through so
    that evaluation can report accuracy as a function of SNR, which is the
    standard way results on this dataset are presented.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    snr_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    snr_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    snr_test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "val": len(self.y_val),
            "test": len(self.y_test),
        }


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the dataset, returning its path.

    Raises:
        FileNotFoundError: if no pickle is found in the downloaded archive.
    """
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    logger.info("Dataset files at %s", root)

    expected = root / DATASET_FILENAME
    if expected.is_file():
        return expected

    # The Kaggle archive has held more than one pickle in the past (10a and
    # 10b). Sorting keeps the fallback deterministic instead of depending on
    # directory iteration order.
    candidates = sorted(root.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl file found under {root}")
    logger.warning("%s not found, falling back to %s", DATASET_FILENAME, candidates[0])
    return candidates[0]


def set_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed every random source this project draws on.

    Args:
        seed: the seed to apply.
        deterministic_cudnn: also force cuDNN into deterministic mode. This
            makes GPU training bit-reproducible at a noticeable cost in speed,
            so it is off by default.

    Note that ``PYTHONHASHSEED`` must be set before the interpreter starts to
    have any effect; setting it here only affects subprocesses.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; only numpy and random were seeded")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Per-worker seed for ``DataLoader(worker_init_fn=...)``.

    Without this each worker process inherits the parent's numpy seed, so any
    numpy-based augmentation would draw the same numbers in every worker.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_raw(path: str | Path | None = None) -> dict[tuple[str, int], np.ndarray]:
    """Load the raw ``(modulation, snr) -> examples`` dict.

    Args:
        path: the pickle to read. If omitted, the dataset is downloaded (or
            taken from the kagglehub cache).
    """
    if path is None:
        path = download_dataset()
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def to_arrays(
    raw: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten the raw dict into parallel arrays.

    Keys are visited in sorted order so that the row order of the output does
    not depend on the insertion order recorded in the pickle.

    Returns:
        ``X`` of shape (N, 2, 128), ``y`` of integer label ids, ``snr`` in dB,
        and the modulation names indexed by label id.
    """
    mods = sorted({mod for mod, _ in raw})
    mod_to_id = {mod: i for i, mod in enumerate(mods)}

    X_parts, y_parts, snr_parts = [], [], []
    for key in sorted(raw):
        mod, snr = key
        examples = raw[key]
        X_parts.append(examples)
        y_parts.append(np.full(len(examples), mod_to_id[mod]))
        snr_parts.append(np.full(len(examples), snr))

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    snr = np.concatenate(snr_parts).astype(np.int64)
    return X, y, snr, mods


def stratification_key(y: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """Map each ``(label, snr)`` pair to a distinct integer id.

    Stratifying on the pair rather than on the label alone keeps every
    class/SNR cell proportionally represented in each partition, so
    accuracy-vs-SNR curves are computed on comparable sample counts.
    """
    pairs = np.stack([y, snr], axis=1)
    _, key = np.unique(pairs, axis=0, return_inverse=True)
    return key


def split(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Partition the dataset, stratified on ``(label, snr)``.

    The default 60/20/20 gives a genuine validation set for model selection,
    which is the sounder protocol. The original paper (O'Shea, Corgan &
    Clancy, 2016) and most follow-up work instead use a single random 50/50
    train/test split with no validation set; pass ``val_size=0.0,
    test_size=0.5`` to reproduce that and make the numbers directly
    comparable with the published accuracy-vs-SNR curves.

    Args:
        X: examples of shape (N, 2, 128).
        y: integer label ids.
        snr: SNR in dB per example.
        val_size: validation fraction of the whole dataset; 0 disables it, in
            which case the validation arrays come back empty.
        test_size: test fraction of the whole dataset.
        seed: seed for the split, independent of any global seeding.

    Raises:
        ValueError: if the requested fractions do not leave a training set.
    """
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must lie in [0, 1), got {val_size}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must lie in (0, 1), got {test_size}")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must leave a training set, got {val_size + test_size}"
        )

    key = stratification_key(y, snr)
    X_fit, X_test, y_fit, y_test, snr_fit, snr_test, key_fit, _ = train_test_split(
        X, y, snr, key, test_size=test_size, random_state=seed, stratify=key
    )

    if val_size == 0.0:
        empty_X = X[:0]
        empty_1d = y[:0]
        return Split(
            X_fit,
            y_fit,
            snr_fit,
            empty_X,
            empty_1d,
            snr[:0],
            X_test,
            y_test,
            snr_test,
        )

    # val_size is a fraction of the whole dataset, so rescale it to a fraction
    # of what is left after the test set was removed.
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_fit,
        y_fit,
        snr_fit,
        test_size=val_fraction,
        random_state=seed,
        stratify=key_fit,
    )
    return Split(
        X_train,
        y_train,
        snr_train,
        X_val,
        y_val,
        snr_val,
        X_test,
        y_test,
        snr_test,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_seed()
    X, y, snr, mods = to_arrays(load_raw())
    logger.info("Loaded %d examples, %d classes: %s", len(X), len(mods), mods)
    logger.info("SNR range: %d to %d dB", snr.min(), snr.max())
    logger.info("Split sizes: %s", split(X, y, snr).sizes())


if __name__ == "__main__":
    main()

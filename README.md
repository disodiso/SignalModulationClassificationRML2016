# Modulation Classification on RadioML 2016.10a

Identify which of 11 modulation schemes produced a radio signal, from 128 raw
I/Q samples, using a 1D convolutional network in PyTorch.

The repository covers two approaches to the same signal, in one notebook:

1. **A classical receiver front-end** (`src/dsp.py`) — root-raised-cosine
   matched filter, Mueller & Muller timing recovery, Costas carrier recovery —
   applied to QPSK to show what the channel impairments actually are and how
   much work it takes to undo them by hand.
2. **A learned classifier** that skips synchronisation entirely and works on
   the raw I/Q, evaluated the way this dataset is normally reported: accuracy
   as a function of SNR.

> **Status.** The exploration (sections 1–6 of the notebook) is complete. The
> classifier (sections 7–12) is scaffolded but not yet implemented.

## The dataset

[RadioML 2016.10a](https://www.deepsig.ai/datasets/), from O'Shea, Corgan &
Clancy, *Convolutional Radio Modulation Recognition Networks* (GNU Radio
Conference, 2016). 220,000 examples: 11 modulations × 20 SNRs from −20 to
+18 dB × 1000 examples, each a 128-sample window of complex baseband (stored
as a `(2, 128)` real array of I and Q).

| | |
|---|---|
| Digital | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, CPFSK, GFSK |
| Analog | AM-DSB, AM-SSB, WBFM |

The windows are cut out of a continuously modulated stream and passed through a
channel model with multipath fading, additive noise, and sample-rate and carrier
frequency offsets. There is no timing or phase reference: symbol timing and
carrier phase are unknown per example, which is what makes the classification
problem non-trivial and what the DSP half of the notebook deals with directly.

The dataset is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is not
redistributed here. It is fetched on first use from
[Kaggle](https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom)
through `kagglehub` and cached under `~/.cache/kagglehub` (about 600 MB).

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"        # or: pip install -r requirements.txt
jupyter lab notebooks/cnn_rml2016.ipynb
```

Requires Python 3.11+. `requirements.txt` pins the exact versions the project
was last run against (Python 3.14.2); the `pyproject.toml` dependencies are
looser ranges for everyday use.

Downloading the dataset needs Kaggle credentials in `~/.kaggle/kaggle.json` or
in `KAGGLE_USERNAME` / `KAGGLE_KEY`. To check the download and the split
protocol without opening the notebook:

```bash
python -m src.data
```

## Layout

```
notebooks/cnn_rml2016.ipynb   the narrative: exploration, DSP, model, results
src/data.py                   download, load, flatten to arrays, stratified split
src/dsp.py                    matched filter, timing recovery, carrier recovery
src/plots.py                  constellation plotting helper
tests/                        pytest suite, no dataset download required
checkpoints/                  model weights (git-ignored)
```

The notebook is narrative only; anything reusable lives in `src/` and is
covered by tests, so the two never drift apart.

```bash
pytest          # 27 tests, runs in ~1s on synthetic signals
ruff check src tests
```

## Notes on the DSP half

The receiver stages are textbook, but two properties of this dataset limit what
they can achieve, and both are stated in the notebook rather than papered over:

* **128 samples is about 16 symbols.** Both the timing and the carrier loop
  need on the order of ten symbols to acquire, so the recovered constellations
  are dominated by the acquisition transient rather than steady-state tracking.
  This is the concrete reason the classifier is fed raw I/Q instead of
  recovered symbols — and the reason published work on this dataset does the
  same.
* **The Costas loop leaves a four-fold phase ambiguity.** Each example settles
  on one of four equally valid quadrant rotations. Harmless for QPSK, which is
  symmetric under it, but it means the recovered symbols carry no absolute
  phase reference.

Two parameters are assumptions rather than measurements: 8 samples per symbol,
and a root-raised-cosine excess bandwidth of 0.35 (the GNU Radio default used
to generate the dataset). Both are constants in `src/dsp.py`.

The decision-directed loops assume a four-quadrant constellation. They are
meaningful for the PSK family and **not** valid for the analog (AM-DSB, AM-SSB,
WBFM) or frequency-shift-keyed (CPFSK, GFSK) modulations in the dataset.

## Evaluation protocol

`src.data.split` stratifies on the `(modulation, SNR)` pair, so every class/SNR
cell is proportionally represented in each partition and accuracy-vs-SNR curves
rest on comparable sample counts.

The default is 60/20/20, which gives a genuine validation set for model
selection. The original paper and most follow-up work instead use a single
random 50/50 train/test split with no validation set; pass `val_size=0.0,
test_size=0.5` to reproduce that protocol when comparing against published
numbers.

## References

* T. J. O'Shea, J. Corgan, T. C. Clancy, *Convolutional Radio Modulation
  Recognition Networks*, GNU Radio Conference, 2016.
* K. Mueller, M. Muller, *Timing Recovery in Digital Synchronous Data
  Receivers*, IEEE Transactions on Communications, 1976.
* M. Rice, *Digital Communications: A Discrete-Time Approach*, for the
  second-order loop filter design used in `dsp.costas_gains`.

## License

Code: MIT, see [LICENSE](LICENSE). The dataset keeps its own CC BY-NC-SA 4.0
terms and is not included in this repository.

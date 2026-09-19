"""Digital receiver front-end used to recover a clean constellation from raw I/Q.

The RadioML 2016.10a records are 128-sample windows cut out of a continuously
modulated stream, so they arrive with unknown symbol timing and an unknown
carrier phase/frequency offset. This module implements the three classical
stages that undo that, in the order a real receiver applies them:

    matched filter  ->  Mueller & Muller timing recovery  ->  Costas loop

Every stage works on a single example at a time, on a complex baseband signal
(``I + jQ``) normalised to unit average power.

Scope and known limitations
---------------------------
* The timing and carrier loops are decision-directed and assume a QPSK-like
  (four-quadrant) constellation. They are meaningful for BPSK/QPSK/8PSK and
  are *not* valid for the analog modulations in the dataset (AM-DSB, AM-SSB,
  WBFM) or for the frequency-shift-keyed ones (CPFSK, GFSK).
* A 128-sample window holds roughly 16 symbols. Both loops need on the order
  of ten symbols to acquire, so what these functions produce is dominated by
  the acquisition transient rather than by steady-state tracking. This is the
  main reason published RML2016 classifiers feed the network raw I/Q instead
  of synchronised symbols; the recovery here is exploratory, not a
  preprocessing step the classifier is expected to depend on.
* The Costas loop leaves the usual four-fold (pi/2) phase ambiguity: each
  example converges to one of four equally valid rotations of the
  constellation, chosen by the noise realisation. The recovered symbols are
  therefore correct up to a quadrant rotation, never absolutely aligned.
"""

from __future__ import annotations

import numpy as np

# Samples per symbol of the digital modulations in RadioML 2016.10a.
DEFAULT_SPS = 8

# Root-raised-cosine excess bandwidth used by the GNU Radio flowgraph that
# generated the dataset. Assumed, not measured from the data.
DEFAULT_ROLLOFF = 0.35


def to_complex(example: np.ndarray) -> np.ndarray:
    """Convert one ``(2, n)`` I/Q example into a complex baseband signal."""
    if example.ndim != 2 or example.shape[0] != 2:
        raise ValueError(f"expected an (2, n) I/Q example, got shape {example.shape}")
    return example[0] + 1j * example[1]


def normalize_power(signal: np.ndarray) -> np.ndarray:
    """Scale a signal to unit average power (a one-shot, ideal AGC).

    The dataset applies random fading, so amplitudes vary by orders of
    magnitude between examples. Normalising per example puts them on a common
    scale, which both the decision-directed loops below and the constellation
    plots rely on.
    """
    power = np.mean(np.abs(signal) ** 2)
    if power == 0:
        return signal
    return signal / np.sqrt(power)


def root_raised_cosine(
    sps: int = DEFAULT_SPS,
    span_symbols: int = 8,
    rolloff: float = DEFAULT_ROLLOFF,
) -> np.ndarray:
    """Unit-energy root-raised-cosine taps, ``span_symbols * sps + 1`` long.

    Args:
        sps: samples per symbol the filter is designed for.
        span_symbols: truncation length of the (infinite) impulse response.
        rolloff: excess-bandwidth factor in [0, 1].
    """
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError(f"rolloff must lie in [0, 1], got {rolloff}")

    n_taps = span_symbols * sps
    t = np.arange(-n_taps / 2, n_taps / 2 + 1) / sps  # time in symbol periods
    beta = rolloff
    h = np.empty_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            # Removable singularity of the closed form at t = +/- T/(4*beta).
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    return h / np.sqrt(np.sum(h**2))


def matched_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a matched filter, keeping the input length and delay alignment.

    Mueller & Muller's timing error detector is derived for a signal that has
    already been through the matched filter; running it on unfiltered samples
    leaves the detector biased by the excess bandwidth.
    """
    return np.convolve(signal, taps, mode="same")


def mueller_muller(
    samples: np.ndarray,
    sps: int = DEFAULT_SPS,
    gain: float = 0.05,
    discard: int = 2,
) -> np.ndarray:
    """Recover symbol timing with a Mueller & Muller synchroniser.

    The loop keeps a fractional sampling instant ``mu`` inside the current
    symbol period and interpolates the input linearly at that instant, so the
    recovered timing is continuous rather than quantised to the 1/sps sample
    grid.

    Args:
        samples: complex baseband signal, unit average power, matched-filtered.
        sps: nominal samples per symbol, the loop's free-running step size.
        gain: loop step size. Larger values acquire faster but leave more
            timing jitter; the step is clipped to keep the loop from stalling
            or running backwards when the error estimate is large.
        discard: number of leading symbols to drop. The first two symbols are
            emitted before the detector has the two-symbol history it needs,
            so their timing is the free-running nominal one rather than a
            corrected estimate.

    Returns:
        The recovered symbols as a complex array.
    """
    if sps < 2:
        raise ValueError(f"sps must be at least 2, got {sps}")
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")

    # Clipping the step guarantees 1 <= step, hence i_in always advances and
    # the loop always terminates, whatever gain the caller passes.
    min_step, max_step = 1.0, 2.0 * sps - 1.0

    symbols: list[complex] = []
    # Hard decisions ("rails"). Pre-padded so that rails[-2] exists from the
    # first iteration; symbols[] carries no such padding.
    rails: list[complex] = [0j, 0j]

    mu = 0.0  # fractional sampling instant inside the current symbol period
    i = 0  # index of the sample just before the interpolation instant

    while i + 1 < len(samples):
        sample = (1.0 - mu) * samples[i] + mu * samples[i + 1]
        rail = np.sign(sample.real) + 1j * np.sign(sample.imag)

        if len(symbols) >= 2:
            # Mueller & Muller timing error detector: the difference of two
            # cross-products between decisions and samples, which is zero on
            # average when sampling at the optimum instant.
            x = (rail - rails[-2]) * np.conj(symbols[-1])
            y = (sample - symbols[-2]) * np.conj(rails[-1])
            error = (y - x).real
        else:
            error = 0.0  # not enough history yet

        symbols.append(sample)
        rails.append(rail)

        mu += float(np.clip(sps + gain * error, min_step, max_step))
        i += int(np.floor(mu))
        mu -= np.floor(mu)

    return np.array(symbols[discard:], dtype=complex)


def costas_gains(loop_bandwidth: float = 0.05, damping: float = np.sqrt(2) / 2):
    """Proportional and integral gains of a second-order loop filter.

    Args:
        loop_bandwidth: natural frequency of the loop, in radians per symbol.
            Relates to the single-sided noise bandwidth by
            ``Bn*T = loop_bandwidth * (damping + 1/(4*damping)) / 2``.
        damping: damping factor; ``sqrt(2)/2`` is the critically damped choice
            that trades acquisition speed against overshoot.

    Returns:
        ``(alpha, beta)``, the gains applied to the phase error and to its
        accumulator respectively.
    """
    if loop_bandwidth <= 0:
        raise ValueError(f"loop_bandwidth must be positive, got {loop_bandwidth}")
    if damping <= 0:
        raise ValueError(f"damping must be positive, got {damping}")

    denominator = 1.0 + 2.0 * damping * loop_bandwidth + loop_bandwidth**2
    alpha = (4.0 * damping * loop_bandwidth) / denominator
    beta = (4.0 * loop_bandwidth**2) / denominator
    return alpha, beta


def costas_loop_qpsk(
    symbols: np.ndarray,
    alpha: float | None = None,
    beta: float | None = None,
) -> np.ndarray:
    """Correct carrier phase and frequency offset on timing-recovered QPSK.

    A decision-directed Costas loop: the error term is the four-quadrant phase
    discriminator, driving a second-order (proportional + integral) filter so
    the loop tracks a constant frequency offset with zero steady-state phase
    error.

    Args:
        symbols: complex symbols, already timing-recovered.
        alpha: proportional gain; defaults to :func:`costas_gains`.
        beta: integral gain; defaults to :func:`costas_gains`.

    Returns:
        The de-rotated symbols, correct up to the four-fold phase ambiguity
        described in the module docstring.
    """
    if alpha is None or beta is None:
        default_alpha, default_beta = costas_gains()
        alpha = default_alpha if alpha is None else alpha
        beta = default_beta if beta is None else beta

    phase = 0.0
    frequency = 0.0
    out = np.zeros(len(symbols), dtype=complex)

    for i, symbol in enumerate(symbols):
        corrected = symbol * np.exp(-1j * phase)
        out[i] = corrected

        error = np.sign(corrected.real) * corrected.imag - np.sign(corrected.imag) * corrected.real

        frequency += beta * error
        phase += frequency + alpha * error
        phase %= 2 * np.pi

    return out


def recover(
    example: np.ndarray,
    sps: int = DEFAULT_SPS,
    gain: float = 0.05,
    rolloff: float | None = DEFAULT_ROLLOFF,
) -> np.ndarray:
    """Run the full front-end on one ``(2, n)`` example.

    Args:
        example: raw I/Q example as stored in the dataset.
        sps: samples per symbol.
        gain: Mueller & Muller loop gain.
        rolloff: matched-filter excess bandwidth, or ``None`` to skip the
            matched filter and feed the timing loop the raw samples.

    Returns:
        The recovered symbols after timing and carrier recovery.
    """
    signal = normalize_power(to_complex(example))
    if rolloff is not None:
        signal = normalize_power(matched_filter(signal, root_raised_cosine(sps, rolloff=rolloff)))
    return costas_loop_qpsk(mueller_muller(signal, sps=sps, gain=gain))

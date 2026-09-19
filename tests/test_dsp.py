"""Checks on the receiver front-end, driven by a synthetic QPSK signal.

Each test builds a transmitter whose symbols are known, pushes it through a
controlled impairment, and asserts that the recovery undoes it. The QPSK
constellation is symmetric under quadrant rotation and the Costas loop has a
matching four-fold ambiguity, so comparisons are made modulo that rotation.
"""

import numpy as np
import pytest

from src import dsp

SPS = 8
N_SYMBOLS = 400


def make_qpsk(n_symbols=N_SYMBOLS, sps=SPS, seed=0):
    """Return (symbols, transmitted pulse-shaped signal at unit power)."""
    rng = np.random.default_rng(seed)
    symbols = (rng.integers(0, 2, n_symbols) * 2 - 1) + 1j * (rng.integers(0, 2, n_symbols) * 2 - 1)
    upsampled = np.zeros(n_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    taps = dsp.root_raised_cosine(sps)
    return symbols, dsp.normalize_power(np.convolve(upsampled, taps, mode="same"))


def receive(transmitted, sps=SPS):
    """Matched-filter a transmitted signal, as a receiver would before timing."""
    taps = dsp.root_raised_cosine(sps)
    return dsp.normalize_power(dsp.matched_filter(transmitted, taps))


def quadrant_error(symbols):
    """Mean distance from the nearest ideal QPSK point, up to a quadrant rotation.

    Symbols are scaled to unit mean magnitude and raised to the fourth power,
    which folds the four quadrants onto the single point -1. The measure is
    therefore blind to the Costas loop's phase ambiguity, and is 0 for an ideal
    constellation and around 2 for one sampled mid-transition.
    """
    normalised = np.asarray(symbols) / np.mean(np.abs(symbols))
    return np.mean(np.abs(normalised**4 + 1.0))


def test_quadrant_error_behaves_as_the_other_tests_assume():
    symbols, _ = make_qpsk(n_symbols=64)

    assert quadrant_error(symbols) == pytest.approx(0.0, abs=1e-9)
    # Blind to amplitude and to the four-fold ambiguity...
    for k in range(4):
        rotated = symbols * 17.0 * np.exp(1j * k * np.pi / 2)
        assert quadrant_error(rotated) == pytest.approx(0.0, abs=1e-9)
    # ...but not to a genuine phase error, which is the point of measuring it.
    assert quadrant_error(symbols * np.exp(1j * 0.3)) > 0.5


def test_root_raised_cosine_is_unit_energy_and_symmetric():
    taps = dsp.root_raised_cosine(SPS, span_symbols=8, rolloff=0.35)
    assert taps.shape == (8 * SPS + 1,)
    assert np.sum(taps**2) == pytest.approx(1.0)
    assert np.allclose(taps, taps[::-1])
    assert np.argmax(taps) == len(taps) // 2


def test_root_raised_cosine_handles_the_removable_singularity():
    # rolloff = 0.25 puts a tap exactly at t = T/(4*beta) = 1 symbol.
    taps = dsp.root_raised_cosine(SPS, rolloff=0.25)
    assert np.all(np.isfinite(taps))


def test_root_raised_cosine_rejects_invalid_rolloff():
    with pytest.raises(ValueError):
        dsp.root_raised_cosine(rolloff=1.5)


def test_to_complex_rejects_wrong_shape():
    with pytest.raises(ValueError):
        dsp.to_complex(np.zeros((3, 128)))


def test_normalize_power_is_idempotent_and_safe_on_silence():
    signal = dsp.normalize_power(np.array([3 + 4j, -1 + 0j, 0 + 2j]))
    assert np.mean(np.abs(signal) ** 2) == pytest.approx(1.0)
    assert np.array_equal(dsp.normalize_power(np.zeros(4, dtype=complex)), np.zeros(4))


def test_mueller_muller_recovers_one_symbol_per_period():
    _, transmitted = make_qpsk()
    signal = receive(transmitted)
    recovered = dsp.mueller_muller(signal, sps=SPS, gain=0.05)
    # Within a few symbols of len(signal)/sps, minus the discarded transient.
    assert abs(len(recovered) - len(signal) // SPS) <= 5


def test_mueller_muller_finds_the_right_sampling_phase():
    """It must land near the best fixed phase, not somewhere between phases."""
    _, transmitted = make_qpsk()
    signal = receive(transmitted)

    by_phase = [quadrant_error(signal[phase::SPS]) for phase in range(SPS)]
    recovered = quadrant_error(dsp.mueller_muller(signal, sps=SPS, gain=0.05))

    assert recovered < min(by_phase) * 1.2
    assert recovered < max(by_phase) / 10


def test_mueller_muller_tracks_a_fractional_timing_offset():
    """A half-sample delay is invisible to any fixed integer sampling phase."""
    _, transmitted = make_qpsk()
    signal = receive(transmitted)
    t = np.arange(len(signal))
    delayed = np.interp(t + 0.5, t, signal.real) + 1j * np.interp(t + 0.5, t, signal.imag)

    recovered = dsp.mueller_muller(delayed, sps=SPS, gain=0.05)
    best_fixed_phase = min(quadrant_error(delayed[phase::SPS]) for phase in range(SPS))

    # No integer sampling phase can reach the true instant; interpolation can.
    assert quadrant_error(recovered[20:]) < best_fixed_phase / 2


def test_mueller_muller_terminates_on_an_absurd_gain():
    """The step clamp must keep the loop advancing whatever the caller passes."""
    _, transmitted = make_qpsk(n_symbols=50)
    recovered = dsp.mueller_muller(receive(transmitted), sps=SPS, gain=1e6)
    assert len(recovered) > 0


def test_mueller_muller_rejects_invalid_parameters():
    _, transmitted = make_qpsk(n_symbols=20)
    with pytest.raises(ValueError):
        dsp.mueller_muller(transmitted, sps=1)
    with pytest.raises(ValueError):
        dsp.mueller_muller(transmitted, gain=0.0)


def test_costas_gains_reproduce_the_standard_loop_filter():
    alpha, beta = dsp.costas_gains(loop_bandwidth=0.05, damping=np.sqrt(2) / 2)
    # The values a hand-tuned implementation would have used.
    assert alpha == pytest.approx(0.132, abs=1e-3)
    assert beta == pytest.approx(0.00932, abs=1e-4)
    # Wider loop bandwidth means more aggressive gains.
    wider_alpha, _ = dsp.costas_gains(loop_bandwidth=0.2)
    assert wider_alpha > alpha


def test_costas_gains_reject_invalid_parameters():
    with pytest.raises(ValueError):
        dsp.costas_gains(loop_bandwidth=0.0)
    with pytest.raises(ValueError):
        dsp.costas_gains(damping=-1.0)


def test_costas_loop_removes_a_frequency_offset():
    symbols, _ = make_qpsk()
    offset = 0.01  # rad/symbol
    rotated = symbols * np.exp(1j * (offset * np.arange(len(symbols)) + 0.7))

    corrected = dsp.costas_loop_qpsk(rotated)

    # Compare after the loop has acquired: a constant frequency offset is
    # exactly what the integral branch is there to drive to zero.
    assert quadrant_error(corrected[100:]) < 1e-3
    assert quadrant_error(rotated[100:]) > 1.0


def test_costas_loop_output_is_correct_up_to_a_quadrant_rotation():
    symbols, _ = make_qpsk(n_symbols=200)
    corrected = dsp.costas_loop_qpsk(symbols * np.exp(1j * 0.4))[50:]
    reference = symbols[50:]

    rotations = [
        np.mean(np.abs(corrected - reference * np.exp(1j * k * np.pi / 2))) for k in range(4)
    ]
    assert min(rotations) < 0.05
    # The loop does not guarantee which of the four it settles on.
    assert max(rotations) > 1.0


def test_recover_chains_the_whole_front_end():
    """Fading gain, carrier offset and unknown timing, undone in one call."""
    _, transmitted = make_qpsk()
    impaired = transmitted * np.exp(1j * (0.002 * np.arange(len(transmitted)) + 1.1))
    example = np.stack([impaired.real, impaired.imag]) * 37.0  # arbitrary fading gain

    recovered = dsp.recover(example, sps=SPS)

    assert len(recovered) > N_SYMBOLS // 2
    assert quadrant_error(recovered[50:]) < quadrant_error(impaired[::SPS]) / 2


def test_recover_can_skip_the_matched_filter():
    _, transmitted = make_qpsk()
    example = np.stack([transmitted.real, transmitted.imag])

    with_filter = quadrant_error(dsp.recover(example, rolloff=0.35)[50:])
    without_filter = quadrant_error(dsp.recover(example, rolloff=None)[50:])

    # The timing error detector is derived for a matched-filtered signal.
    assert with_filter < without_filter

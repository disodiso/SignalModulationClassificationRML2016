"""Constellation plotting helpers shared by the exploration notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def constellation(
    ax: plt.Axes,
    series: dict[str, np.ndarray],
    title: str = "",
    alpha: float = 0.4,
    limit_from: str | None = None,
) -> plt.Axes:
    """Draw one or more complex series on a square I/Q plane.

    Args:
        ax: the axes to draw on.
        series: label -> complex array. Drawing order is insertion order, so
            put the densest cloud first and it ends up underneath.
        title: axes title.
        alpha: marker opacity; lower it when overlaying dense clouds of
            oversampled points, raise it for sparse symbol-rate ones.
        limit_from: which series sets the axis limits; defaults to the first.
            Worth setting when one series is much wider than the others and
            would otherwise squeeze them into the centre.
    """
    markers = ["x", "x", "o", "^", "s"]
    # markers is longer than any series dict we pass, so zip stops at the data.
    for (label, values), marker in zip(series.items(), markers, strict=False):
        ax.scatter(
            np.real(values),
            np.imag(values),
            marker=marker,
            alpha=alpha,
            label=label,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    reference = series[limit_from] if limit_from else next(iter(series.values()))
    limit = np.max(np.abs(np.asarray(reference))) * 1.1 if len(reference) else 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    if len(series) > 1:
        ax.legend(loc="upper right", fontsize="small")
    return ax

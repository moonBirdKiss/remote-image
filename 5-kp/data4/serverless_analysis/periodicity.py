"""Periodicity detection and artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(slots=True)
class PeriodicityResult:
    """Artifacts and summary statistics for a periodic function."""

    detected_period_bins: int
    detected_period_seconds: float
    spectral_power_ratio: float
    spectral_peak_to_median: float
    acf_peak: float
    acf_second_peak: float
    request_count: int
    bin_count: int
    counts_series: pd.Series
    acf_lags: np.ndarray
    acf_values: np.ndarray
    frequencies: np.ndarray
    powers: np.ndarray


def build_count_series(bin_counter: dict[int, int]) -> pd.Series:
    """Expand sparse bin counts into a dense time series."""

    if not bin_counter:
        return pd.Series(dtype="int64")

    index = np.arange(min(bin_counter), max(bin_counter) + 1, dtype=np.int64)
    values = np.zeros(index.shape[0], dtype=np.int64)
    for bin_id, count in bin_counter.items():
        values[bin_id - index[0]] = count
    return pd.Series(values, index=index, name="request_count")


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized autocorrelation for non-negative lags."""

    sample_count = len(values)
    if sample_count == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if max_lag is None:
        max_lag = sample_count - 1
    max_lag = min(max_lag, sample_count - 1)

    centered = values - np.mean(values)
    variance = np.var(centered)
    if variance <= 0:
        lags = np.arange(max_lag + 1, dtype=int)
        acf = np.zeros(max_lag + 1, dtype=float)
        if acf.size:
            acf[0] = 1.0
        return lags, acf

    full = np.correlate(centered, centered, mode="full")
    positive = full[sample_count - 1 : sample_count + max_lag]
    acf = positive / positive[0]
    lags = np.arange(max_lag + 1, dtype=int)
    return lags, acf


def detect_periodicity(
    bin_counter: dict[int, int],
    *,
    bin_size_seconds: int,
    min_bins: int,
    min_requests: int,
    power_ratio_threshold: float,
    peak_to_median_threshold: float,
    acf_threshold: float,
    second_acf_threshold: float,
    min_cycles: int = 3,
) -> PeriodicityResult | None:
    """Detect periodicity from per-bin request counts.

    The series is first evaluated with a periodogram. The dominant spectral
    period must then align with strong positive autocorrelation at one and two
    multiples of that period.
    """

    counts_series = build_count_series(bin_counter)
    if counts_series.empty:
        return None

    values = counts_series.to_numpy(dtype=float)
    bin_count = len(values)
    request_count = int(values.sum())
    if bin_count < min_bins or request_count < min_requests:
        return None

    centered = values - values.mean()
    if np.allclose(centered, 0.0):
        return None

    frequencies, powers = signal.periodogram(
        centered,
        fs=1.0,
        detrend="constant",
        window="hann",
        scaling="spectrum",
    )
    mask = frequencies > 0
    frequencies = frequencies[mask]
    powers = powers[mask]
    if frequencies.size == 0 or not np.any(powers > 0):
        return None

    periods = 1.0 / frequencies
    max_period_bins = bin_count / float(min_cycles)
    mask = (periods >= 2.0) & (periods <= max_period_bins)
    frequencies = frequencies[mask]
    powers = powers[mask]
    periods = periods[mask]
    if frequencies.size == 0:
        return None

    dominant_index = int(np.argmax(powers))
    dominant_period_bins = int(round(periods[dominant_index]))
    dominant_power = float(powers[dominant_index])
    spectral_power_ratio = dominant_power / float(np.sum(powers))
    spectral_peak_to_median = dominant_power / float(np.median(powers) + 1e-12)

    max_lag = min(bin_count - 1, dominant_period_bins * 3)
    lags, acf_values = autocorrelation(values, max_lag=max_lag)
    if lags.size == 0 or dominant_period_bins >= acf_values.size:
        return None

    search_radius = max(1, int(round(dominant_period_bins * 0.1)))
    first_start = max(1, dominant_period_bins - search_radius)
    first_end = min(acf_values.size - 1, dominant_period_bins + search_radius)
    first_window = acf_values[first_start : first_end + 1]
    acf_peak = float(np.max(first_window))
    first_peak_lag = first_start + int(np.argmax(first_window))

    second_target = first_peak_lag * 2
    if second_target < acf_values.size:
        second_start = max(1, second_target - search_radius)
        second_end = min(acf_values.size - 1, second_target + search_radius)
        second_window = acf_values[second_start : second_end + 1]
        acf_second_peak = float(np.max(second_window))
    else:
        acf_second_peak = 0.0

    if spectral_power_ratio < power_ratio_threshold:
        return None
    if spectral_peak_to_median < peak_to_median_threshold:
        return None
    if acf_peak < acf_threshold:
        return None
    if second_target < acf_values.size and acf_second_peak < second_acf_threshold:
        return None

    return PeriodicityResult(
        detected_period_bins=first_peak_lag,
        detected_period_seconds=first_peak_lag * float(bin_size_seconds),
        spectral_power_ratio=spectral_power_ratio,
        spectral_peak_to_median=spectral_peak_to_median,
        acf_peak=acf_peak,
        acf_second_peak=acf_second_peak,
        request_count=request_count,
        bin_count=bin_count,
        counts_series=counts_series,
        acf_lags=lags,
        acf_values=acf_values,
        frequencies=frequencies,
        powers=powers,
    )


def save_periodicity_artifacts(
    output_dir: Path,
    result: PeriodicityResult,
    *,
    bin_size_seconds: int,
) -> None:
    """Write periodicity evidence CSVs, a report, and a plot."""

    output_dir.mkdir(parents=True, exist_ok=True)

    counts_df = pd.DataFrame(
        {
            "time_bin_id": result.counts_series.index.astype(np.int64),
            "time_bin_start_seconds": result.counts_series.index.astype(np.int64)
            * int(bin_size_seconds),
            "request_count": result.counts_series.to_numpy(dtype=int),
        }
    )
    counts_df.to_csv(output_dir / "binned_counts.csv", index=False)

    acf_df = pd.DataFrame(
        {
            "lag_bins": result.acf_lags,
            "lag_seconds": result.acf_lags * int(bin_size_seconds),
            "autocorrelation": result.acf_values,
        }
    )
    acf_df.to_csv(output_dir / "acf.csv", index=False)

    periodogram_df = pd.DataFrame(
        {
            "frequency_per_bin": result.frequencies,
            "period_bins": 1.0 / result.frequencies,
            "period_seconds": (1.0 / result.frequencies) * int(bin_size_seconds),
            "power": result.powers,
        }
    ).sort_values("period_seconds")
    periodogram_df.to_csv(output_dir / "periodogram.csv", index=False)

    report_lines = [
        f"detected_period_bins: {result.detected_period_bins}",
        f"detected_period_seconds: {result.detected_period_seconds:.3f}",
        f"detected_period_minutes: {result.detected_period_seconds / 60.0:.3f}",
        f"request_count: {result.request_count}",
        f"bin_count: {result.bin_count}",
        f"spectral_power_ratio: {result.spectral_power_ratio:.6f}",
        f"spectral_peak_to_median: {result.spectral_peak_to_median:.6f}",
        f"acf_peak: {result.acf_peak:.6f}",
        f"acf_second_peak: {result.acf_second_peak:.6f}",
    ]
    (output_dir / "periodicity_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    time_minutes = (
        result.counts_series.index.to_numpy(dtype=float) * float(bin_size_seconds) / 60.0
    )
    axes[0].plot(time_minutes, result.counts_series.to_numpy(dtype=float), color="#005f73")
    axes[0].set_title("Binned Request Counts")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Requests / bin")

    lag_minutes = result.acf_lags * float(bin_size_seconds) / 60.0
    axes[1].plot(lag_minutes, result.acf_values, color="#9b2226")
    axes[1].axvline(
        result.detected_period_seconds / 60.0,
        color="#ee9b00",
        linestyle="--",
        label=f"Detected period: {result.detected_period_seconds / 60.0:.2f} min",
    )
    axes[1].set_title("Autocorrelation")
    axes[1].set_xlabel("Lag (minutes)")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].legend()

    period_minutes = (1.0 / result.frequencies) * float(bin_size_seconds) / 60.0
    order = np.argsort(period_minutes)
    axes[2].plot(period_minutes[order], result.powers[order], color="#3a5a40")
    axes[2].axvline(
        result.detected_period_seconds / 60.0,
        color="#bc6c25",
        linestyle="--",
        label="Detected period",
    )
    axes[2].set_title("Periodogram")
    axes[2].set_xlabel("Candidate period (minutes)")
    axes[2].set_ylabel("Spectral power")
    axes[2].legend()

    figure.savefig(output_dir / "periodicity.png", dpi=160)
    plt.close(figure)

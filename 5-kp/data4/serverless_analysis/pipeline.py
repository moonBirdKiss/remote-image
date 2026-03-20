"""Main analysis pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import (
    EXPORT_COLUMNS,
    READ_DTYPES,
    list_csv_files,
    normalize_chunk,
    sanitize_function_name,
    validate_columns,
)
from .periodicity import detect_periodicity, save_periodicity_artifacts


@dataclass(slots=True)
class AnalysisConfig:
    """Runtime configuration for the analysis job."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    timestamp_column: str = "time_worker"
    chunk_size: int = 200_000
    bin_size_seconds: int = 10
    min_bins: int = 60
    min_requests: int = 15
    power_ratio_threshold: float = 0.18
    peak_to_median_threshold: float = 5.0
    acf_threshold: float = 0.35
    second_acf_threshold: float = 0.15
    max_files: int | None = None


@dataclass(slots=True)
class FunctionSummary:
    """Compact per-function metadata tracked during streaming."""

    func_name: str
    safe_name: str
    request_count: int = 0
    min_timestamp: float = np.inf
    max_timestamp: float = -np.inf

    @property
    def duration_seconds(self) -> float:
        """Return the observed request span in seconds."""
        if not np.isfinite(self.min_timestamp) or not np.isfinite(self.max_timestamp):
            return 0.0
        return float(self.max_timestamp - self.min_timestamp)


@dataclass(slots=True)
class AnalysisSummary:
    """Top-level summary returned by the pipeline."""

    total_files: int
    total_rows: int
    total_functions: int
    periodic_functions: int
    single_output_dir: Path
    period_output_dir: Path


def run_analysis(config: AnalysisConfig) -> AnalysisSummary:
    """Run the full analysis workflow."""

    csv_files = list_csv_files(config.data_dir)
    if config.max_files is not None:
        csv_files = csv_files[: config.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {config.data_dir}")

    validate_columns(csv_files[0])

    single_root = config.output_dir / "single"
    period_root = config.output_dir / "period"
    single_root.mkdir(parents=True, exist_ok=True)
    period_root.mkdir(parents=True, exist_ok=True)

    assigned_names: dict[str, str] = {}
    function_to_safe_name: dict[str, str] = {}
    summaries: dict[str, FunctionSummary] = {}
    binned_counts: dict[str, Counter[int]] = defaultdict(Counter)
    total_rows = 0

    for csv_path in csv_files:
        for chunk in pd.read_csv(
            csv_path,
            usecols=list(EXPORT_COLUMNS),
            dtype=READ_DTYPES,
            chunksize=config.chunk_size,
        ):
            normalized = normalize_chunk(chunk, config.timestamp_column)
            if normalized.empty:
                continue

            total_rows += len(normalized)
            _register_functions(normalized["funcName"].unique(), assigned_names, function_to_safe_name)
            _append_single_function_rows(normalized, single_root, function_to_safe_name, config.timestamp_column)
            _update_summaries(normalized, summaries, function_to_safe_name, config.timestamp_column)
            _update_binned_counts(
                normalized,
                binned_counts,
                config.timestamp_column,
                config.bin_size_seconds,
            )

    _finalize_single_outputs(single_root, summaries, config.timestamp_column)
    periodic_count = _detect_periodic_functions(
        binned_counts=binned_counts,
        function_to_safe_name=function_to_safe_name,
        summaries=summaries,
        period_root=period_root,
        config=config,
    )
    _write_function_index(config.output_dir, summaries, function_to_safe_name)

    return AnalysisSummary(
        total_files=len(csv_files),
        total_rows=total_rows,
        total_functions=len(summaries),
        periodic_functions=periodic_count,
        single_output_dir=single_root,
        period_output_dir=period_root,
    )


def _register_functions(
    func_names: np.ndarray,
    assigned_names: dict[str, str],
    function_to_safe_name: dict[str, str],
) -> None:
    """Assign output directory names for functions discovered in a chunk."""

    for func_name in func_names:
        if func_name not in function_to_safe_name:
            function_to_safe_name[func_name] = sanitize_function_name(func_name, assigned_names)


def _append_single_function_rows(
    chunk: pd.DataFrame,
    single_root: Path,
    function_to_safe_name: dict[str, str],
    timestamp_column: str,
) -> None:
    """Append request rows into per-function CSV files."""

    for func_name, func_df in chunk.groupby("funcName", sort=False):
        safe_name = function_to_safe_name[func_name]
        function_dir = single_root / safe_name
        function_dir.mkdir(parents=True, exist_ok=True)
        output_path = function_dir / "requests.csv"

        export_df = func_df.loc[:, EXPORT_COLUMNS].sort_values(timestamp_column, kind="stable")
        write_header = not output_path.exists()
        export_df.to_csv(output_path, mode="a", header=write_header, index=False)


def _update_summaries(
    chunk: pd.DataFrame,
    summaries: dict[str, FunctionSummary],
    function_to_safe_name: dict[str, str],
    timestamp_column: str,
) -> None:
    """Update per-function counts and time ranges."""

    for func_name, func_df in chunk.groupby("funcName", sort=False):
        summary = summaries.get(func_name)
        if summary is None:
            summary = FunctionSummary(
                func_name=func_name,
                safe_name=function_to_safe_name[func_name],
            )
            summaries[func_name] = summary

        timestamps = func_df[timestamp_column]
        summary.request_count += int(len(func_df))
        summary.min_timestamp = min(summary.min_timestamp, float(timestamps.min()))
        summary.max_timestamp = max(summary.max_timestamp, float(timestamps.max()))


def _update_binned_counts(
    chunk: pd.DataFrame,
    binned_counts: dict[str, Counter[int]],
    timestamp_column: str,
    bin_size_seconds: int,
) -> None:
    """Aggregate per-function request counts per time bin."""

    time_bins = np.floor(
        chunk[timestamp_column].to_numpy(dtype=float) / float(bin_size_seconds)
    ).astype(np.int64)
    aggregated = (
        pd.DataFrame(
            {
                "funcName": chunk["funcName"].to_numpy(copy=False),
                "time_bin": time_bins,
            }
        )
        .groupby(["funcName", "time_bin"])
        .size()
    )
    for (func_name, time_bin), count in aggregated.items():
        binned_counts[func_name][int(time_bin)] += int(count)


def _finalize_single_outputs(
    single_root: Path,
    summaries: dict[str, FunctionSummary],
    timestamp_column: str,
) -> None:
    """Sort per-function CSVs and write summary metadata."""

    summary_rows = []
    for summary in sorted(summaries.values(), key=lambda item: item.func_name):
        function_dir = single_root / summary.safe_name
        requests_path = function_dir / "requests.csv"
        requests_df = pd.read_csv(requests_path, dtype=READ_DTYPES)
        requests_df[timestamp_column] = pd.to_numeric(
            requests_df[timestamp_column],
            errors="coerce",
        )
        requests_df = requests_df.sort_values(timestamp_column, kind="stable")
        requests_df.to_csv(requests_path, index=False)

        summary_text = "\n".join(
            [
                f"funcName: {summary.func_name}",
                f"safe_name: {summary.safe_name}",
                f"request_count: {summary.request_count}",
                f"min_{timestamp_column}: {summary.min_timestamp:.6f}",
                f"max_{timestamp_column}: {summary.max_timestamp:.6f}",
                f"duration_seconds: {summary.duration_seconds:.6f}",
            ]
        )
        (function_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")

        summary_rows.append(
            {
                "funcName": summary.func_name,
                "safe_name": summary.safe_name,
                "request_count": summary.request_count,
                f"min_{timestamp_column}": summary.min_timestamp,
                f"max_{timestamp_column}": summary.max_timestamp,
                "duration_seconds": summary.duration_seconds,
            }
        )

    pd.DataFrame(summary_rows).to_csv(single_root / "function_summary.csv", index=False)


def _detect_periodic_functions(
    *,
    binned_counts: dict[str, Counter[int]],
    function_to_safe_name: dict[str, str],
    summaries: dict[str, FunctionSummary],
    period_root: Path,
    config: AnalysisConfig,
) -> int:
    """Run periodicity detection and save evidence for periodic functions."""

    summary_rows = []
    periodic_count = 0

    for func_name in sorted(binned_counts):
        result = detect_periodicity(
            binned_counts[func_name],
            bin_size_seconds=config.bin_size_seconds,
            min_bins=config.min_bins,
            min_requests=config.min_requests,
            power_ratio_threshold=config.power_ratio_threshold,
            peak_to_median_threshold=config.peak_to_median_threshold,
            acf_threshold=config.acf_threshold,
            second_acf_threshold=config.second_acf_threshold,
        )
        if result is None:
            continue

        periodic_count += 1
        safe_name = function_to_safe_name[func_name]
        output_dir = period_root / safe_name
        save_periodicity_artifacts(
            output_dir,
            result,
            bin_size_seconds=config.bin_size_seconds,
        )
        summary_rows.append(
            {
                "funcName": func_name,
                "safe_name": safe_name,
                "request_count": result.request_count,
                "bin_count": result.bin_count,
                "detected_period_bins": result.detected_period_bins,
                "detected_period_seconds": result.detected_period_seconds,
                "detected_period_minutes": result.detected_period_seconds / 60.0,
                "spectral_power_ratio": result.spectral_power_ratio,
                "spectral_peak_to_median": result.spectral_peak_to_median,
                "acf_peak": result.acf_peak,
                "acf_second_peak": result.acf_second_peak,
                "duration_seconds": summaries[func_name].duration_seconds,
            }
        )

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(
            ["detected_period_seconds", "funcName"]
        ).to_csv(period_root / "periodic_functions.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "funcName",
                "safe_name",
                "request_count",
                "bin_count",
                "detected_period_bins",
                "detected_period_seconds",
                "detected_period_minutes",
                "spectral_power_ratio",
                "spectral_peak_to_median",
                "acf_peak",
                "acf_second_peak",
                "duration_seconds",
            ]
        ).to_csv(period_root / "periodic_functions.csv", index=False)

    return periodic_count


def _write_function_index(
    output_dir: Path,
    summaries: dict[str, FunctionSummary],
    function_to_safe_name: dict[str, str],
) -> None:
    """Write a global function-to-output mapping."""

    index_rows = []
    for func_name in sorted(function_to_safe_name):
        safe_name = function_to_safe_name[func_name]
        summary = summaries[func_name]
        index_rows.append(
            {
                "funcName": func_name,
                "safe_name": safe_name,
                "request_count": summary.request_count,
                "single_dir": f"single/{safe_name}",
                "period_dir": f"period/{safe_name}",
            }
        )

    pd.DataFrame(index_rows).to_csv(output_dir / "function_index.csv", index=False)

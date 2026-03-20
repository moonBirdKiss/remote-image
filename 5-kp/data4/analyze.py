#!/usr/bin/env python3
"""Command-line entry point for serverless request analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from serverless_analysis.pipeline import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze serverless request logs, export per-function request series, "
            "and detect strongly periodic request patterns."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input CSV files. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for analysis outputs. Default: output",
    )
    parser.add_argument(
        "--timestamp-column",
        choices=("time_worker", "time_frontend"),
        default="time_worker",
        help="Timestamp column used for time-series analysis. Default: time_worker",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per pandas chunk while streaming CSV files. Default: 200000",
    )
    parser.add_argument(
        "--bin-size-seconds",
        type=int,
        default=10,
        help="Bin width in seconds for periodicity detection. Default: 10",
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=60,
        help="Minimum number of time bins required before testing periodicity. Default: 60",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=15,
        help="Minimum number of requests required before testing periodicity. Default: 15",
    )
    parser.add_argument(
        "--power-ratio-threshold",
        type=float,
        default=0.18,
        help="Minimum dominant spectral power ratio for a periodic candidate. Default: 0.18",
    )
    parser.add_argument(
        "--peak-to-median-threshold",
        type=float,
        default=5.0,
        help="Minimum periodogram peak-to-median ratio for a periodic candidate. Default: 5.0",
    )
    parser.add_argument(
        "--acf-threshold",
        type=float,
        default=0.35,
        help="Minimum autocorrelation peak near the detected period. Default: 0.35",
    )
    parser.add_argument(
        "--second-acf-threshold",
        type=float,
        default=0.15,
        help="Minimum autocorrelation near twice the detected period. Default: 0.15",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for the number of CSV files to process, useful for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the analysis from CLI arguments."""
    args = build_parser().parse_args(argv)
    config = AnalysisConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timestamp_column=args.timestamp_column,
        chunk_size=args.chunk_size,
        bin_size_seconds=args.bin_size_seconds,
        min_bins=args.min_bins,
        min_requests=args.min_requests,
        power_ratio_threshold=args.power_ratio_threshold,
        peak_to_median_threshold=args.peak_to_median_threshold,
        acf_threshold=args.acf_threshold,
        second_acf_threshold=args.second_acf_threshold,
        max_files=args.max_files,
    )

    try:
        summary = run_analysis(config)
    except Exception as exc:  # pragma: no cover - top-level UX path
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Completed analysis: "
        f"{summary.total_files} file(s), "
        f"{summary.total_rows:,} row(s), "
        f"{summary.total_functions} function(s), "
        f"{summary.periodic_functions} periodic function(s)."
    )
    print(f"Per-function output: {summary.single_output_dir}")
    print(f"Periodicity output: {summary.period_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

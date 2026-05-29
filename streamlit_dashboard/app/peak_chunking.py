"""
Metadata-driven spectral chunking for NMR metabolomics pipelines.

Builds ppm windows from metadata peak positions (±half_width), merges
overlapping windows, and produces a boolean mask for fitting subsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class ChunkingResult:
    """Output of the metadata-based chunking step."""

    metadata_peaks: list[float]
    raw_intervals: list[tuple[float, float]]
    merged_chunks: list[tuple[float, float]]
    mask: np.ndarray
    stats: dict = field(default_factory=dict)
    metadata_peak_table: pd.DataFrame | None = None

    @property
    def detected_peaks(self) -> list[float]:
        """Backward-compatible alias for metadata peak centers."""
        return self.metadata_peaks


def label_from_metadata_column(meta_df: pd.DataFrame, column: str, fallback: str) -> str:
    """Read a display label from a metadata column, or return fallback if missing/empty."""
    if column not in meta_df.columns:
        return fallback
    return _clean_meta_label(meta_df[column].iloc[0], fallback)


def _clean_meta_label(value, fallback: str) -> str:
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return fallback
    return text


def build_metadata_peak_table(
    positions: list[float] | None,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a table with peak name and ppm (sorted high → low ppm)."""
    records = metadata_peak_records(positions, names)
    if not records:
        return pd.DataFrame(columns=["peak_name", "peak_ppm"])
    df = pd.DataFrame([{"peak_name": name, "peak_ppm": ppm} for name, ppm in records])
    return df.sort_values("peak_ppm", ascending=False).reset_index(drop=True)


def metadata_peak_records(
    positions: list[float] | None,
    names: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Return (peak_name, ppm) pairs in metadata order."""
    if not positions:
        return []
    records: list[tuple[str, float]] = []
    for i, p in enumerate(positions):
        if p is None or not np.isfinite(p):
            continue
        if names and i < len(names):
            label = _clean_meta_label(names[i], f"Peak {len(records) + 1}")
        else:
            label = f"Peak {len(records) + 1}"
        records.append((label, float(p)))
    return records


def extract_metadata_peaks(positions: list[float] | None) -> list[float]:
    """
    Return finite ppm centers from metadata (substrate, metabolites, water).

    Args:
        positions: Peak centers already parsed from metadata (e.g. via extract_ppm_all).
    """
    if not positions:
        return []
    peaks: list[float] = []
    for p in positions:
        if p is not None and np.isfinite(p):
            peaks.append(float(p))
    return sorted(peaks)


def generate_chunk_windows(
    peaks: list[float],
    half_width: float,
    ppm_min: float,
    ppm_max: float,
) -> list[tuple[float, float]]:
    """Build [center - half_width, center + half_width] intervals clipped to the spectrum."""
    intervals = []
    for peak in peaks:
        low = max(ppm_min, peak - half_width)
        high = min(ppm_max, peak + half_width)
        if low < high:
            intervals.append((float(low), float(high)))
    return intervals


def merge_overlapping_chunks(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Merge overlapping or touching ppm intervals into non-overlapping chunks.

    Example:
        Peak A chunk: 2.5–3.5 ppm, Peak B chunk: 2.9–3.9 ppm → merged 2.5–3.9 ppm.
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [list(sorted_intervals[0])]

    for low, high in sorted_intervals[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])

    return [(m[0], m[1]) for m in merged]


# Backward-compatible alias
merge_overlapping_intervals = merge_overlapping_chunks
peaks_to_intervals = generate_chunk_windows


def intervals_to_mask(ppm: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    """Build a boolean row mask from merged ppm intervals (works for ascending or descending ppm)."""
    ppm = np.asarray(ppm, dtype=float)
    mask = np.zeros(len(ppm), dtype=bool)
    for low, high in intervals:
        lo, hi = min(low, high), max(low, high)
        mask |= (ppm >= lo) & (ppm <= hi)
    return mask


def apply_metadata_chunking(
    df: pd.DataFrame,
    metadata_peaks: list[float] | None = None,
    metadata_peak_names: list[str] | None = None,
    half_width: float = 0.5,
) -> ChunkingResult:
    """
    Build chunk windows from metadata ppm positions → merge → mask for fitting.

    Args:
        df: Spectrum DataFrame (column 0 = ppm, remaining columns = time frames).
        metadata_peaks: Ppm centers from metadata (substrate, Metabolite_*_ppm, water).
        metadata_peak_names: Labels aligned with metadata_peaks (e.g. metabolite names).
        half_width: Half-width of each chunk in ppm (default 0.5 → ±0.5 ppm window).

    Returns:
        ChunkingResult with mask, merged chunks, and diagnostic stats.
    """
    ppm = df.iloc[:, 0].to_numpy()
    ppm_min = float(np.nanmin(ppm))
    ppm_max = float(np.nanmax(ppm))

    peak_table = build_metadata_peak_table(metadata_peaks, metadata_peak_names)
    peak_centers = peak_table["peak_ppm"].tolist() if not peak_table.empty else extract_metadata_peaks(metadata_peaks)
    raw_intervals = generate_chunk_windows(peak_centers, half_width, ppm_min, ppm_max)
    merged_chunks = merge_overlapping_chunks(raw_intervals)

    if not merged_chunks:
        merged_chunks = [(ppm_min, ppm_max)]
        mask = np.ones(len(ppm), dtype=bool)
    else:
        mask = intervals_to_mask(ppm, merged_chunks)

    n_raw = len(raw_intervals)
    n_merged = len(merged_chunks)

    stats = {
        "n_metadata_peaks": len(peak_centers),
        "n_peak_centers_for_chunking": len(peak_centers),
        "n_raw_chunks": n_raw,
        "n_merged_chunks": n_merged,
        "n_merged_from_overlap": max(0, n_raw - n_merged),
        "fraction_of_rows_in_chunks": float(mask.sum()) / len(mask) if len(mask) else 0.0,
        "half_width_ppm": half_width,
        "chunk_source": "metadata",
    }

    return ChunkingResult(
        metadata_peaks=peak_centers,
        raw_intervals=raw_intervals,
        merged_chunks=merged_chunks,
        mask=mask,
        stats=stats,
        metadata_peak_table=peak_table if not peak_table.empty else None,
    )


def apply_peak_chunking(
    df: pd.DataFrame,
    metadata_peaks: list[float] | None = None,
    metadata_peak_names: list[str] | None = None,
    half_width: float = 0.5,
    **_ignored,
) -> ChunkingResult:
    """Backward-compatible entry point; ignores legacy peak-detection kwargs."""
    return apply_metadata_chunking(
        df,
        metadata_peaks=metadata_peaks,
        metadata_peak_names=metadata_peak_names,
        half_width=half_width,
    )


def save_processing_timings(
    output_dir: Union[str, Path],
    timings: dict,
) -> None:
    """Save processing step durations (seconds) to processing_timings.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([timings]).to_csv(out / "processing_timings.csv", index=False)


def save_chunk_artifacts(
    output_dir: Union[str, Path],
    result: ChunkingResult,
    half_width: float,
    enable_chunking: bool = True,
    timings: dict | None = None,
) -> None:
    """
    Persist chunk boundaries and summary stats for the visualization panel.

    Writes under output_dir:
        - chunking_summary.csv
        - chunk_regions.csv
        - metadata_peaks.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "enabled": enable_chunking,
        "half_width_ppm": half_width,
        **result.stats,
    }
    pd.DataFrame([summary]).to_csv(out / "chunking_summary.csv", index=False)

    if timings:
        save_processing_timings(out, timings)

    if result.merged_chunks:
        chunk_rows = [
            {
                "chunk_id": i + 1,
                "ppm_low": lo,
                "ppm_high": hi,
                "width_ppm": hi - lo,
            }
            for i, (lo, hi) in enumerate(result.merged_chunks)
        ]
        pd.DataFrame(chunk_rows).to_csv(out / "chunk_regions.csv", index=False)
    else:
        pd.DataFrame(columns=["chunk_id", "ppm_low", "ppm_high", "width_ppm"]).to_csv(
            out / "chunk_regions.csv", index=False
        )

    if result.metadata_peak_table is not None and not result.metadata_peak_table.empty:
        peak_df = result.metadata_peak_table
    elif result.metadata_peaks:
        peak_df = build_metadata_peak_table(result.metadata_peaks)
    else:
        peak_df = pd.DataFrame(columns=["peak_name", "peak_ppm"])

    peak_df.to_csv(out / "metadata_peaks.csv", index=False)
    if "peak_ppm" in peak_df.columns:
        peak_df[["peak_ppm"]].to_csv(out / "detected_peaks.csv", index=False)
    else:
        pd.DataFrame(columns=["peak_ppm"]).to_csv(out / "detected_peaks.csv", index=False)


def log_chunking_stats(result: ChunkingResult, prefix: str = "[Metadata chunking]") -> None:
    """Print chunking diagnostics to stdout."""
    s = result.stats
    print(
        f"{prefix} metadata peaks: {s.get('n_metadata_peaks', 0)}, "
        f"raw chunks: {s.get('n_raw_chunks', 0)}, "
        f"merged chunks: {s.get('n_merged_chunks', 0)} "
        f"({s.get('n_merged_from_overlap', 0)} merges from overlap), "
        f"rows used: {s.get('fraction_of_rows_in_chunks', 0):.1%}"
    )

"""
Panel 7: Visualize metadata-driven fitting chunks on the NMR spectrum.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from peak_chunking import build_metadata_peak_table


def _apply_nmr_ppm_axis(fig: go.Figure) -> go.Figure:
    """NMR convention: chemical shift decreases left to right (high ppm on the left)."""
    fig.update_xaxes(autorange="reversed")
    return fig


def _downsample_xy(ppm: pd.Series, intensity: pd.Series, max_points: int) -> tuple[pd.Series, pd.Series]:
    n = len(ppm)
    if n <= max_points:
        return ppm, intensity
    step = max(1, n // max_points)
    return ppm.iloc[::step], intensity.iloc[::step]


def _peaks_in_chunk(
    metadata_peaks: pd.DataFrame | None,
    lo: float,
    hi: float,
) -> list[str]:
    if metadata_peaks is None or metadata_peaks.empty or "peak_ppm" not in metadata_peaks.columns:
        return []
    in_chunk = metadata_peaks[
        (metadata_peaks["peak_ppm"] >= lo) & (metadata_peaks["peak_ppm"] <= hi)
    ]
    if "peak_name" in in_chunk.columns:
        names = [str(n).strip() for n in in_chunk["peak_name"] if str(n).strip()]
        return names
    return [f"{float(p):.2f} ppm" for p in in_chunk["peak_ppm"]]


def _chunk_label(
    metadata_peaks: pd.DataFrame | None,
    lo: float,
    hi: float,
    chunk_id: int,
) -> str:
    names = _peaks_in_chunk(metadata_peaks, lo, hi)
    if names:
        return ", ".join(names)
    return f"Chunk {chunk_id}"


def plot_overview_spectrum(
    ppm: pd.Series,
    intensity: pd.Series,
    chunks_df: pd.DataFrame,
    metadata_peaks: pd.DataFrame | None = None,
    *,
    label: str = "Spectrum",
    max_points: int = 5000,
) -> go.Figure:
    """Full spectrum with shaded chunk regions labeled by peak name(s)."""
    colors = px.colors.qualitative.Set2
    ppm_ds, y_ds = _downsample_xy(ppm, intensity, max_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ppm_ds,
            y=y_ds,
            mode="lines",
            name=label,
            line=dict(color="#1f77b4", width=1.2),
        )
    )

    if chunks_df is not None and not chunks_df.empty:
        for i, row in chunks_df.iterrows():
            lo = float(row["ppm_low"])
            hi = float(row["ppm_high"])
            cid = int(row.get("chunk_id", i + 1))
            color = colors[(cid - 1) % len(colors)]
            region_label = _chunk_label(metadata_peaks, lo, hi, cid)
            fig.add_vrect(
                x0=lo,
                x1=hi,
                fillcolor=color,
                opacity=0.22,
                line_width=1,
                line_color=color,
                annotation_text=region_label,
                annotation_position="top left",
            )

    if metadata_peaks is not None and not metadata_peaks.empty:
        for _, peak_row in metadata_peaks.iterrows():
            ppm_val = float(peak_row["peak_ppm"])
            name = str(peak_row.get("peak_name", "") or "").strip()
            fig.add_vline(
                x=ppm_val,
                line_dash="dot",
                line_color="crimson",
                line_width=1,
                opacity=0.75,
                annotation_text=name or f"{ppm_val:.2f}",
                annotation_position="top",
            )

    fig.update_layout(
        title="Full spectrum — chunk regions",
        xaxis_title="Chemical shift (ppm)",
        yaxis_title="Intensity (a.u.)",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _apply_nmr_ppm_axis(fig)


def render_chunked_spectra(
    ppm: pd.Series,
    intensity: pd.Series,
    chunks_df: pd.DataFrame,
    metadata_peaks: pd.DataFrame | None = None,
    *,
    label: str = "Spectrum",
    max_points: int = 5000,
    pad_fraction: float = 0.05,
) -> list[go.Figure]:
    """
    Build one Plotly figure per merged chunk (no full-spectrum render).

    Args:
        ppm: Chemical shift axis.
        intensity: Spectrum intensities for one frame or mean.
        chunks_df: DataFrame with chunk_id, ppm_low, ppm_high.
        metadata_peaks: Optional DataFrame with peak_ppm column for vertical markers.
    """
    if chunks_df is None or chunks_df.empty:
        return []

    colors = px.colors.qualitative.Set2
    figures: list[go.Figure] = []

    for i, row in chunks_df.iterrows():
        lo = float(row["ppm_low"])
        hi = float(row["ppm_high"])
        pad = pad_fraction * (hi - lo) if hi > lo else 0.1
        cid = int(row.get("chunk_id", i + 1))
        region_label = _chunk_label(metadata_peaks, lo, hi, cid)

        mask = (ppm >= lo - pad) & (ppm <= hi + pad)
        ppm_z, y_z = _downsample_xy(ppm[mask], intensity[mask], max_points)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ppm_z,
                y=y_z,
                mode="lines",
                name=label,
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        color = colors[(cid - 1) % len(colors)]
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor=color,
            opacity=0.35,
            line_width=2,
            line_color=color,
            annotation_text=f"{region_label} ({hi:.3f}–{lo:.3f} ppm)",
            annotation_position="top left",
        )

        if metadata_peaks is not None and not metadata_peaks.empty:
            in_chunk = metadata_peaks[
                (metadata_peaks["peak_ppm"] >= lo) & (metadata_peaks["peak_ppm"] <= hi)
            ]
            for _, peak_row in in_chunk.iterrows():
                ppm_val = float(peak_row["peak_ppm"])
                name = str(peak_row.get("peak_name", "") or "").strip()
                fig.add_vline(
                    x=ppm_val,
                    line_dash="dot",
                    line_color="crimson",
                    annotation_text=name or f"{ppm_val:.2f} ppm",
                    annotation_position="top",
                )

        fig.update_layout(
            title=f"{region_label} ({hi:.3f}–{lo:.3f} ppm)",
            xaxis_title="Chemical shift (ppm)",
            yaxis_title="Intensity (a.u.)",
            template="plotly_white",
            height=420,
        )
        _apply_nmr_ppm_axis(fig)
        figures.append(fig)

    return figures


class ChunkRegionsPlot:
    """
    Per-chunk views of merged fitting regions from metadata ppm windows.

    Reads chunk_regions.csv / metadata_peaks.csv from the standard output folder.
    """

    CHUNK_COLORS = px.colors.qualitative.Set2

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.output_dir = Path("output", f"{self.file_name}_output")
        self.plot_dir = Path(self.output_dir, "plots")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.template = "plotly_white"
        self.max_points = 5000

        self.data = self._load_spectrum(file_path)
        self.chunks_df = self._load_optional_csv("chunk_regions.csv")
        self.metadata_df = self._load_metadata_peaks()
        self.summary_df = self._load_optional_csv("chunking_summary.csv")
        self.timings_df = self._load_optional_csv("processing_timings.csv")

    def _load_metadata_peaks(self) -> pd.DataFrame | None:
        for name in ("metadata_peaks.csv", "detected_peaks.csv"):
            df = self._load_optional_csv(name)
            if df is not None and not df.empty:
                return df
        return None

    def _load_optional_csv(self, name: str) -> pd.DataFrame | None:
        path = self.output_dir / name
        if path.exists():
            return pd.read_csv(path)
        return None

    def _load_spectrum(self, fp: str) -> pd.DataFrame:
        df = pd.read_csv(fp, sep=None, engine="python", on_bad_lines="skip")
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="any")
        if not df.empty:
            df = df.rename(columns={df.columns[0]: "chemical_shift_ppm"})
        return df

    @property
    def chunking_enabled(self) -> bool:
        if self.summary_df is None or self.summary_df.empty:
            return self.chunks_df is not None and not self.chunks_df.empty
        if "enabled" in self.summary_df.columns:
            return bool(self.summary_df["enabled"].iloc[0])
        return True

    def n_frames(self) -> int:
        return max(self.data.shape[1] - 1, 0)

    def _spectrum_xy(self, frame: int, use_mean: bool = False):
        ppm = self.data["chemical_shift_ppm"]
        time_cols = list(self.data.columns[1:])
        n_frames = len(time_cols)
        if use_mean:
            y = self.data.iloc[:, 1:].mean(axis=1)
            label = f"Mean of {n_frames} frames"
        else:
            col_idx = min(max(frame, 1), n_frames) if n_frames else 1
            col_name = time_cols[col_idx - 1] if time_cols else f"frame {col_idx}"
            y = self.data.iloc[:, col_idx]
            label = f"Frame {col_idx}/{n_frames} ({col_name})"
        return ppm, y, label

    def plot_chunk_detail(self, chunk_id: int, frame: int = 1, use_mean: bool = True) -> go.Figure:
        """Zoomed view of a single merged chunk region."""
        if self.chunks_df is None or self.chunks_df.empty:
            return go.Figure().update_layout(title="No chunks available")

        figures = self.render_all_chunks(frame=frame, use_mean=use_mean)
        for fig in figures:
            if fig.layout.title.text and fig.layout.title.text.startswith(f"Chunk {chunk_id}"):
                return fig
        return figures[0] if figures else go.Figure().update_layout(title="No chunks available")

    def _metadata_for_plots(self) -> pd.DataFrame | None:
        display = self.metadata_peaks_display_df()
        if display is not None and not display.empty:
            return display
        if self.metadata_df is not None and not self.metadata_df.empty:
            return self.metadata_df
        return None

    def plot_overview(self, frame: int = 1, use_mean: bool = True) -> go.Figure | None:
        """Full spectrum with chunk regions labeled by peak name(s)."""
        if self.chunks_df is None or self.chunks_df.empty:
            return None
        ppm, y, label = self._spectrum_xy(frame, use_mean=use_mean)
        return plot_overview_spectrum(
            ppm,
            y,
            self.chunks_df,
            self._metadata_for_plots(),
            label=label,
            max_points=self.max_points,
        )

    def render_all_chunks(self, frame: int = 1, use_mean: bool = True) -> list[go.Figure]:
        """Return one figure per merged chunk (metadata-driven windows)."""
        if self.chunks_df is None or self.chunks_df.empty:
            return []
        ppm, y, label = self._spectrum_xy(frame, use_mean=use_mean)
        return render_chunked_spectra(
            ppm,
            y,
            self.chunks_df,
            self._metadata_for_plots(),
            label=label,
            max_points=self.max_points,
        )

    def chunks_table(self) -> pd.DataFrame | None:
        return self.chunks_df

    def metadata_peaks_display_df(self) -> pd.DataFrame | None:
        """Peak table with name and ppm (high → low); upgrades legacy ppm-only CSVs."""
        if self.metadata_df is None or self.metadata_df.empty:
            return None
        df = self.metadata_df.copy()
        if "peak_name" not in df.columns and "peak_ppm" in df.columns:
            df = build_metadata_peak_table(df["peak_ppm"].tolist())
        if "peak_ppm" in df.columns:
            df = df.sort_values("peak_ppm", ascending=False).reset_index(drop=True)
        cols = [c for c in ("peak_name", "peak_ppm") if c in df.columns]
        return df[cols] if cols else df

    def summary_metrics(self) -> dict:
        if self.summary_df is None or self.summary_df.empty:
            n_chunks = 0 if self.chunks_df is None else len(self.chunks_df)
            metrics = {"n_merged_chunks": n_chunks}
        else:
            metrics = self.summary_df.iloc[0].to_dict()
        if self.timings_df is not None and not self.timings_df.empty:
            metrics.update(self.timings_df.iloc[0].to_dict())
        return metrics

    def save_fig(self, fig, name):
        """Save plotly figure as PDF and PNG (Kaleido)."""
        pio.write_image(fig, f"{name}.pdf", format="pdf", engine="kaleido", width=1200, height=800)
        pio.write_image(fig, f"{name}.png", format="png", engine="kaleido", width=1200, height=800)

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class StackedSpectraPlot:
    """
    Creates stacked/waterfall visualizations of NMR spectra over time.

    Supports:
    - legacy Plotly stacked plot
    - legacy Matplotlib 2D stacked plot
    - new Matplotlib 3D waterfall plot with raw/fitted/diff modes
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.output_dir = Path("output", self.file_name + "_output")
        self.plot_dir = Path(self.output_dir, "plots")
        self.max_points = 1200

        self.data = self._load_clean_csv(file_path)
        self.x = self.data.iloc[:, 0]
        self.n_timepoints = self.data.shape[1] - 1

        sum_fit_path = Path(self.output_dir, "sum_fit.csv")
        diff_path = Path(self.output_dir, "differences.csv")

        self.sum_data = self._load_clean_csv(sum_fit_path) if sum_fit_path.exists() else None
        self.diff_data = self._load_clean_csv(diff_path) if diff_path.exists() else None

    def _load_clean_csv(self, fp):
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

    def _downsample_xy(self, x_series, y_series):
        n = len(x_series)
        if n <= self.max_points:
            return x_series, y_series

        step = max(1, n // self.max_points)
        return x_series.iloc[::step], y_series.iloc[::step]

    def _smooth_signal(self, y_values, window_size=5):
        if window_size is None or window_size <= 1:
            return y_values

        window_size = int(window_size)
        if window_size % 2 == 0:
            window_size += 1

        if len(y_values) < window_size:
            return y_values

        kernel = np.ones(window_size) / window_size
        return np.convolve(y_values, kernel, mode="same")

    def validate_peak_positions(self):
        peak_positions = []

        for t in range(1, self.n_timepoints + 1):
            y = self.data.iloc[:, t].values
            idx = np.argmax(y)
            peak_positions.append(self.x.iloc[idx])

        if not peak_positions:
            return []

        peak_positions = np.array(peak_positions)

        return [
            float(np.min(peak_positions)),
            float(np.median(peak_positions)),
            float(np.max(peak_positions)),
        ]

    def plot_matplotlib_3d(
        self,
        azim=45,
        elev=25,
        plot_mode="Raw only",
        time_start=1,
        time_end=None,
        show_every_nth=1,
        smooth=True,
        smooth_window=5,
    ):
        plt.rcParams["font.family"] = "Arial"

        if time_end is None:
            time_end = self.n_timepoints

        time_start = max(1, int(time_start))
        time_end = min(self.n_timepoints, int(time_end))
        show_every_nth = max(1, int(show_every_nth))

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")

        timepoints = list(range(time_start, time_end + 1, show_every_nth))

        if not timepoints:
            ax.text2D(0.4, 0.5, "No spectra selected", transform=ax.transAxes)
            return fig

        colors = plt.cm.rainbow(np.linspace(0, 1, len(timepoints)))

        for idx, t in enumerate(timepoints):
            y_position = idx + 1

            if plot_mode == "Raw only":
                x_data = self.x
                z_data = self.data.iloc[:, t]

                x_data, z_data = self._downsample_xy(x_data, z_data)
                z_vals = z_data.values.astype(float)

                if smooth:
                    z_vals = self._smooth_signal(z_vals, smooth_window)

                ax.plot(
                    x_data.values,
                    np.full(len(x_data), y_position),
                    z_vals,
                    color=colors[idx],
                    linewidth=1.2,
                    alpha=0.95,
                )

            elif plot_mode == "Fitted only":
                if self.sum_data is not None and t < self.sum_data.shape[1]:
                    x_fit = self.sum_data.iloc[:, 0]
                    z_fit = self.sum_data.iloc[:, t]

                    x_fit, z_fit = self._downsample_xy(x_fit, z_fit)
                    z_vals = z_fit.values.astype(float)

                    if smooth:
                        z_vals = self._smooth_signal(z_vals, smooth_window)

                    ax.plot(
                        x_fit.values,
                        np.full(len(x_fit), y_position),
                        z_vals,
                        color=colors[idx],
                        linewidth=1.4,
                        alpha=0.95,
                    )

            elif plot_mode == "Raw + fitted":
                x_data = self.x
                z_data = self.data.iloc[:, t]

                x_data, z_data = self._downsample_xy(x_data, z_data)
                z_vals = z_data.values.astype(float)

                if smooth:
                    z_vals = self._smooth_signal(z_vals, smooth_window)

                ax.plot(
                    x_data.values,
                    np.full(len(x_data), y_position),
                    z_vals,
                    color=colors[idx],
                    linewidth=1.1,
                    alpha=0.95,
                )

                if self.sum_data is not None and t < self.sum_data.shape[1]:
                    x_fit = self.sum_data.iloc[:, 0]
                    z_fit = self.sum_data.iloc[:, t]

                    x_fit, z_fit = self._downsample_xy(x_fit, z_fit)
                    z_fit_vals = z_fit.values.astype(float)

                    if smooth:
                        z_fit_vals = self._smooth_signal(z_fit_vals, smooth_window)

                    ax.plot(
                        x_fit.values,
                        np.full(len(x_fit), y_position),
                        z_fit_vals,
                        color="black",
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.8,
                    )

            elif plot_mode == "Diff only":
                if self.diff_data is not None and t < self.diff_data.shape[1]:
                    x_diff = self.diff_data.iloc[:, 0]
                    z_diff = self.diff_data.iloc[:, t]

                    x_diff, z_diff = self._downsample_xy(x_diff, z_diff)
                    z_vals = z_diff.values.astype(float)

                    if smooth:
                        z_vals = self._smooth_signal(z_vals, smooth_window)

                    ax.plot(
                        x_diff.values,
                        np.full(len(x_diff), y_position),
                        z_vals,
                        color=colors[idx],
                        linewidth=1.2,
                        alpha=0.95,
                    )

        ax.set_xlabel("Chemical Shift (ppm)", labelpad=14, fontname="Arial")
        ax.set_ylabel("Time Point", labelpad=14, fontname="Arial")
        ax.set_zlabel("Intensity", labelpad=12, fontname="Arial")
        ax.set_title("Waterfall Plot", pad=20, fontname="Arial")

        ax.set_xlim(self.x.max(), self.x.min())

        y_positions = np.arange(1, len(timepoints) + 1)
        if len(timepoints) > 12:
            step = max(1, len(timepoints) // 10)
            ax.set_yticks(y_positions[::step])
            ax.set_yticklabels([str(tp) for tp in timepoints[::step]], fontname="Arial")
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([str(tp) for tp in timepoints], fontname="Arial")

        for label in ax.get_xticklabels():
            label.set_fontname("Arial")
        for label in ax.get_yticklabels():
            label.set_fontname("Arial")
        for label in ax.get_zticklabels():
            label.set_fontname("Arial")

        ax.xaxis._axinfo["grid"]["linewidth"] = 0.8
        ax.yaxis._axinfo["grid"]["linewidth"] = 0.0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0.8

        ax.view_init(elev=elev, azim=azim)

        try:
            ax.set_box_aspect((3.2, 1.5, 1.4))
        except Exception:
            pass

        plt.tight_layout()
        return fig

    def plot_plotly(self, spacing=None, show_fit=False, show_every_nth=1):
        fig = go.Figure()

        if spacing is None:
            max_intensity = self.data.iloc[:, 1:].max().max()
            spacing = max_intensity * 1.5

        timepoints = range(1, self.n_timepoints + 1, show_every_nth)

        for idx, t in enumerate(timepoints):
            x_data = self.x
            y_data = self.data.iloc[:, t]
            x_data, y_data = self._downsample_xy(x_data, y_data)

            offset = idx * spacing

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data + offset,
                    mode="lines",
                    name=f"Time {t}",
                    line=dict(width=1),
                    hovertemplate="ppm: %{x:.2f}<br>Intensity: %{y:.2f}<extra></extra>",
                )
            )

            if show_fit and self.sum_data is not None and t < self.sum_data.shape[1]:
                x_fit = self.sum_data.iloc[:, 0]
                y_fit = self.sum_data.iloc[:, t]
                x_fit, y_fit = self._downsample_xy(x_fit, y_fit)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit + offset,
                        mode="lines",
                        name=f"Fit {t}",
                        line=dict(width=2, dash="dash"),
                        hovertemplate="ppm: %{x:.2f}<br>Fit: %{y:.2f}<extra></extra>",
                    )
                )

        fig.update_layout(
            title="Stacked NMR Spectra (Waterfall Plot)",
            xaxis_title="Chemical Shift (ppm)",
            yaxis_title="Intensity (stacked)",
            template="plotly_white",
            height=800,
            hovermode="closest",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig.update_xaxes(autorange="reversed")
        return fig

    def plot_matplotlib(self, spacing=None, show_fit=False, show_every_nth=1, colormap="viridis"):
        fig, ax = plt.subplots(figsize=(12, 10))

        if spacing is None:
            max_intensity = self.data.iloc[:, 1:].max().max()
            spacing = max_intensity * 1.5

        timepoints = range(1, self.n_timepoints + 1, show_every_nth)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(timepoints)))

        for idx, t in enumerate(timepoints):
            y_data = self.data.iloc[:, t].values
            offset = idx * spacing

            ax.plot(
                self.x,
                y_data + offset,
                color=colors[idx],
                linewidth=0.8,
                label=f"Time {t}",
            )

            if show_fit and self.sum_data is not None and t < self.sum_data.shape[1]:
                y_fit = self.sum_data.iloc[:, t].values
                ax.plot(
                    self.x,
                    y_fit + offset,
                    color=colors[idx],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )

        ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
        ax.set_ylabel("Intensity (stacked)", fontsize=12)
        ax.set_title("Stacked NMR Spectra (Waterfall Plot)", fontsize=14)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        return fig

    def save_fig(self, fig, name):
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

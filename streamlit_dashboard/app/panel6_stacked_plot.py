import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StackedSpectraPlot:
    """
    3D waterfall plot for time-resolved NMR spectra.

    X-axis: Chemical Shift (ppm)
    Y-axis: Time point
    Z-axis: Intensity
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
        if sum_fit_path.exists():
            self.sum_data = self._load_clean_csv(sum_fit_path)
        else:
            self.sum_data = None

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
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "chemical_shift_ppm"})

        return df

    def _downsample_xy(self, x_series, y_series):
        n = len(x_series)
        if n <= self.max_points:
            return x_series, y_series

        step = max(1, n // self.max_points)
        return x_series.iloc[::step], y_series.iloc[::step]

    def plot_matplotlib_3d(self, azim=30, elev=25, show_fit=False, show_every_nth=1):
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")

        timepoints = list(range(1, self.n_timepoints + 1, show_every_nth))
        colors = plt.cm.turbo(np.linspace(0, 1, len(timepoints)))

        for idx, t in enumerate(timepoints):
            y_idx = idx + 1

            x_data = self.x
            z_data = self.data.iloc[:, t]
            x_data, z_data = self._downsample_xy(x_data, z_data)

            ax.plot(
                x_data.values,
                np.full(len(x_data), y_idx),
                z_data.values,
                color=colors[idx],
                linewidth=1.0,
                alpha=0.95
            )

            if show_fit and self.sum_data is not None and t < self.sum_data.shape[1]:
                x_fit = self.sum_data.iloc[:, 0]
                z_fit = self.sum_data.iloc[:, t]
                x_fit, z_fit = self._downsample_xy(x_fit, z_fit)

                ax.plot(
                    x_fit.values,
                    np.full(len(x_fit), y_idx),
                    z_fit.values,
                    color="black",
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.7
                )

        ax.set_xlabel("Chemical Shift (ppm)", labelpad=12)
        ax.set_ylabel("Time Point", labelpad=12)
        ax.set_zlabel("Intensity", labelpad=12)
        ax.set_title("3D Waterfall Plot", pad=20)

        ax.set_xlim(self.x.max(), self.x.min())

        y_positions = np.arange(1, len(timepoints) + 1)

        if len(timepoints) > 15:
            step = max(1, len(timepoints) // 10)
            ax.set_yticks(y_positions[::step])
            ax.set_yticklabels([str(tp) for tp in timepoints[::step]])
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([str(tp) for tp in timepoints])

        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)

        try:
            ax.set_box_aspect((3.0, 1.4, 1.2))
        except Exception:
            pass

        plt.tight_layout()
        return fig

    def save_fig(self, fig, name):
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
import os
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class StackedSpectraPlot:
<<<<<<< HEAD
    """
    Creates stacked (waterfall) visualizations of NMR spectra over time.

    This class loads a time-resolved NMR spectrum from a CSV file and provides
    interactive (Plotly) and static (Matplotlib) stacked spectra plots. Each
    timepoint is plotted with a vertical offset so that spectral changes over
    time become visually apparent. Optionally, fitted spectra (sum fits) can
    be overlaid if a corresponding `sum_fit.csv` file is available.

    Attributes:
        file_path (str): Path to the original spectrum CSV file.
        file_name (str): Base name of the spectrum file (without extension).
        output_dir (Path): Output directory for derived files of this dataset.
        plot_dir (Path): Directory where plots can be stored.
        max_points (int): Maximum number of points per trace sent to the browser
            (used to downsample for performance).
        data (pandas.DataFrame): Cleaned raw spectrum data, with the first
            column being chemical shift ("chemical_shift_ppm") and remaining
            columns being timepoints.
        x (pandas.Series): Chemical shift axis (ppm) extracted from `data`.
        n_timepoints (int): Number of timepoints (i.e. number of spectrum
            columns minus the x-axis column).
        sum_data (pandas.DataFrame or None): Optional sum-fit data loaded from
            `sum_fit.csv`, cleaned in the same way as `data`. If the file is
            not present, this attribute is set to None.

    Methods:
        __init__(file_path):
            Loads and cleans the spectral data and (if available) the sum-fit data.
        
        plot_plotly(spacing=None, show_fit=False, show_every_nth=1):
            Creates an interactive stacked spectra (waterfall) plot using Plotly.

        plot_matplotlib(spacing=None, show_fit=False, show_every_nth=1, colormap='viridis'):
            Creates a Matplotlib-based stacked spectra plot (suitable for PDF export).

        save_fig(fig, name):
            Saves a Matplotlib figure as a PDF file in the plots directory.

        _downsample_xy(x_series, y_series):
            Internal helper to downsample data for performance.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.output_dir = Path('output', self.file_name + '_output')
        self.plot_dir = Path(self.output_dir, 'plots')
        self.max_points = 5000  # cap points per trace sent to browser
        
        # Load data robustly (delimiter sniffing, bad-line skip, decimal commas)
        self.data = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
        for col in self.data.columns:
            if self.data[col].dtype == object:
                self.data[col] = self.data[col].astype(str).str.replace(",", ".", regex=False)
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data = self.data.dropna(axis=1, how="all")
        self.data = self.data.dropna(axis=0, how="any")
        # Ensure first column has a stable name
        shift_col = self.data.columns[0]
        self.data = self.data.rename(columns={shift_col: "chemical_shift_ppm"})
        self.x = self.data.iloc[:, 0]  # Chemical shifts
        self.n_timepoints = self.data.shape[1] - 1
        
        # Optional: Load fitted spectra
        sum_fit_path = Path(self.output_dir, 'sum_fit.csv')
        if sum_fit_path.exists():
            self.sum_data = pd.read_csv(sum_fit_path, sep=None, engine="python", on_bad_lines="skip")
            for col in self.sum_data.columns:
                if self.sum_data[col].dtype == object:
                    self.sum_data[col] = self.sum_data[col].astype(str).str.replace(",", ".", regex=False)
                self.sum_data[col] = pd.to_numeric(self.sum_data[col], errors="coerce")
            self.sum_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.sum_data = self.sum_data.dropna(axis=1, how="all")
            self.sum_data = self.sum_data.dropna(axis=0, how="any")
            shift_col = self.sum_data.columns[0]
            self.sum_data = self.sum_data.rename(columns={shift_col: "chemical_shift_ppm"})
        else:
            self.sum_data = None

=======
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

>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
    def _downsample_xy(self, x_series, y_series):
        """Reduce payload size for plotting."""
        n = len(x_series)
        if n <= self.max_points:
            return x_series, y_series
        step = max(1, n // self.max_points)
        return x_series.iloc[::step], y_series.iloc[::step]
    
    def plot_plotly(self, spacing=None, show_fit=False, show_every_nth=1):
        """
        Creates an interactive stacked NMR spectra (waterfall) plot using Plotly.

<<<<<<< HEAD
        Each selected timepoint is plotted with a vertical offset so that temporal
        changes in the spectra can be compared visually. Optionally, the corresponding
        sum-fit spectrum can be overlaid if `sum_data` is available.

        Args:
            spacing (float, optional):
                Vertical offset between successive spectra. If None, a default
                value based on the global maximum intensity is used.
            show_fit (bool, optional):
                If True and `sum_data` is not None, the fitted spectra from
                `sum_fit.csv` are plotted alongside the raw spectra (dashed lines).
            show_every_nth (int, optional):
                Only every n-th timepoint is plotted (e.g. 1 = all, 2 = every 2nd, ...).
=======
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
        smooth_window=5
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
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)

        Returns:
            plotly.graph_objects.Figure: The generated Plotly figure.
        """
        fig = go.Figure()
        
        # Calculate automatic spacing
        if spacing is None:
            max_intensity = self.data.iloc[:, 1:].max().max()
            spacing = max_intensity * 1.5
        
        # Choose timepoints
        timepoints = range(1, self.n_timepoints + 1, show_every_nth)
        
        for idx, t in enumerate(timepoints):
<<<<<<< HEAD
            col_idx = t  # Columnindex (first column is x)
            x_data = self.x
            y_data = self.data.iloc[:, col_idx]
            x_data, y_data = self._downsample_xy(x_data, y_data)
            
            # Vertikal Offset
            offset = idx * spacing
            
            # Original-Spectrum
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data + offset,
                mode='lines',
                name=f'Time {t}',
                line=dict(width=1),
                hovertemplate='ppm: %{x:.2f}<br>Intensity: %{y:.2f}<extra></extra>'
            ))
            
            # Fitted Spectrum (optional)
            if show_fit and self.sum_data is not None:
                x_fit = self.sum_data.iloc[:,0]
                y_fit = self.sum_data.iloc[:, col_idx]
                x_fit, y_fit = self._downsample_xy(x_fit, y_fit)
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit + offset,
                    mode='lines',
                    name=f'Fit {t}',
                    line=dict(width=2, dash='dash'),
                    hovertemplate='ppm: %{x:.2f}<br>Fit: %{y:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Stacked NMR Spectra (Waterfall Plot)',
            xaxis_title='Chemical Shift (ppm)',
            yaxis_title='Intensity (stacked)',
            template='plotly_white',
            height=800,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Reverse x-axis (NMR convention)
        fig.update_xaxes(autorange='reversed')
        
        return fig
    
    def plot_matplotlib(self, spacing=None, show_fit=False, show_every_nth=1, 
                       colormap='viridis'):
        """
        Creates a static stacked spectra (waterfall) plot using Matplotlib.

        This version is intended primarily for high-quality exports (e.g. PDF).
        It supports the same options for spacing, fit overlay and timepoint
        subsampling as `plot_plotly`.

        Args:
            spacing (float, optional):
                Vertical offset between successive spectra. If None, a default
                value based on the global maximum intensity is used.
            show_fit (bool, optional):
                If True and `sum_data` is not None, fitted spectra are plotted
                as dashed lines over the raw spectra.
            show_every_nth (int, optional):
                Only every n-th timepoint is plotted.
            colormap (str, optional):
                Name of a Matplotlib colormap used to color the time series.

        Returns:
            matplotlib.figure.Figure: The generated Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Automatical Spacing
        if spacing is None:
            max_intensity = self.data.iloc[:, 1:].max().max()
            spacing = max_intensity * 1.5
        
        # Colormap
        timepoints = range(1, self.n_timepoints + 1, show_every_nth)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(timepoints)))
        
        for idx, t in enumerate(timepoints):
            col_idx = t
            y_data = self.data.iloc[:, col_idx].values
            offset = idx * spacing
            
            # Original-Spectrum
            ax.plot(self.x, y_data + offset, 
                   color=colors[idx], 
                   linewidth=0.8, 
                   label=f'Time {t}')
            
            # Fitted Spectrum (optional)
            if show_fit and self.sum_data is not None:
                y_fit = self.sum_data.iloc[:, col_idx].values
                ax.plot(self.x, y_fit + offset, 
                       color=colors[idx], 
                       linewidth=1.5, 
                       linestyle='--',
                       alpha=0.7)
        
        ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_ylabel('Intensity (stacked)', fontsize=12)
        ax.set_title('Stacked NMR Spectra (Waterfall Plot)', fontsize=14)
        ax.invert_xaxis()  # NMR convention
        ax.grid(True, alpha=0.3)
        
        # Legend out of Plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
=======
            y_position = idx + 1

            # ---------------- Raw only ----------------
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
                    alpha=0.95
                )

            # ---------------- Fitted only ----------------
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
                        alpha=0.95
                    )

            # ---------------- Raw + fitted ----------------
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
                    alpha=0.95
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
                        alpha=0.8
                    )

            # ---------------- Difference / residual only ----------------
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
                        alpha=0.95
                    )

        ax.set_xlabel("Chemical Shift (ppm)", labelpad=14, fontname="Arial")
        ax.set_ylabel("Time Point", labelpad=14, fontname="Arial")
        ax.set_zlabel("Intensity", labelpad=12, fontname="Arial")
        ax.set_title("Waterfall Plot", pad=20, fontname="Arial")

        # NMR convention
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

        # Keep x-axis grid lines, remove y-axis grid lines
        ax.xaxis._axinfo["grid"]["linewidth"] = 0.8
        ax.yaxis._axinfo["grid"]["linewidth"] = 0.0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0.8

        ax.view_init(elev=elev, azim=azim)

        try:
            ax.set_box_aspect((3.2, 1.5, 1.4))
        except Exception:
            pass

>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
        plt.tight_layout()
        return fig
    
    def save_fig(self, fig, name):
        """
        Saves a Matplotlib figure as a PDF file in the plots directory.

        The file is saved with a `.pdf` extension and a resolution of 300 dpi.
        The plot directory is created if it does not yet exist.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            name (str or Path): Base path/name for the file. The method appends
                the `.pdf` suffix automatically.

        Returns:
            None
        """
        os.makedirs(self.plot_dir, exist_ok=True)
<<<<<<< HEAD
        fig.savefig(f'{name}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
=======
        fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import peak_fitting_v6


class Reference:
    """
    Handle reference spectrum fitting and conversion of kinetics values to mmol.

    This version is robust against dynamic kinetics column names.
    Older code expected fixed columns such as ReacSubs, Metab1 and Water.
    The current pipeline writes real metabolite/substrate names into kinetics.csv,
    so save_kinetics_mmol() now converts every numeric kinetics column except Time_Step.
    """

    def __init__(self, fp_ref, fp_meta, fp_data):
        self.fp_ref = fp_ref
        self.fp_meta = fp_meta
        self.fp_data = fp_data

        self.data = self._read_csv_flexible(fp_ref)
        self.chem_shifts = self.data.iloc[:, 0]

        self.file_name_ref = os.path.splitext(os.path.basename(fp_ref))[0]
        self.file_name = os.path.splitext(os.path.basename(fp_data))[0]

        self.plot_dir = Path("output", self.file_name_ref + "_output", "plots")
        self.reference_pdf = Path(self.plot_dir, f"Reference_{self.file_name_ref}")

        self.output_dir = Path("output", self.file_name + "_output")
        self.kin_fp = Path(self.output_dir, "kinetics.csv")

        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.kin_fp.exists():
            raise FileNotFoundError(f"Kinetics file not found: {self.kin_fp}")

        self.kin_df = self._read_csv_flexible(self.kin_fp)

        self.LorentzianFit = peak_fitting_v6.PeakFitting(
            fp_file=fp_ref,
            fp_meta=fp_meta,
            enable_chunking=False,
        )
        self.fitting_params = self.LorentzianFit.fit(save_csv=False)
        self.reference_value = self.ReferenceValue()

    @staticmethod
    def _read_csv_flexible(path):
        """Read comma/semicolon CSVs and normalize numeric columns."""
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                converted = pd.to_numeric(df[col], errors="ignore")
                df[col] = converted

        return df

    @staticmethod
    def _first_existing_column(df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _water_amp_column(self):
        """Find the fitted water amplitude column, usually Water_amp_4.7."""
        amp_cols = [
            col for col in self.fitting_params.columns
            if "water" in str(col).lower() and "_amp_" in str(col).lower()
        ]

        if amp_cols:
            return amp_cols[0]

        # fallback: any amplitude column if water label is different
        any_amp_cols = [
            col for col in self.fitting_params.columns
            if "_amp_" in str(col).lower()
        ]
        if any_amp_cols:
            return any_amp_cols[-1]

        raise KeyError(
            "No water amplitude column found in fitting_params. "
            f"Available columns: {self.fitting_params.columns.tolist()}"
        )

    def _water_pos_width_amp_columns(self):
        """Find water position, width and amplitude columns for plotting."""
        water_pos_cols = [
            col for col in self.fitting_params.columns
            if "water" in str(col).lower() and "_pos_" in str(col).lower()
        ]
        water_width_cols = [
            col for col in self.fitting_params.columns
            if "water" in str(col).lower() and "_width_" in str(col).lower()
        ]
        water_amp_cols = [
            col for col in self.fitting_params.columns
            if "water" in str(col).lower() and "_amp_" in str(col).lower()
        ]

        if water_pos_cols and water_width_cols and water_amp_cols:
            return water_pos_cols[0], water_width_cols[0], water_amp_cols[0]

        # fallback: use matching position/width/amplitude columns by order
        pos_cols = [col for col in self.fitting_params.columns if "_pos_" in str(col).lower()]
        width_cols = [col for col in self.fitting_params.columns if "_width_" in str(col).lower()]
        amp_cols = [col for col in self.fitting_params.columns if "_amp_" in str(col).lower()]

        if pos_cols and width_cols and amp_cols:
            return pos_cols[-1], width_cols[-1], amp_cols[-1]

        raise KeyError(
            "Could not find position/width/amplitude columns for reference plot. "
            f"Available columns: {self.fitting_params.columns.tolist()}"
        )

    def ReferenceValue(self):
        """
        Calculate conversion factor from fitted water amplitude to mmol.
        """
        meta = self.LorentzianFit.meta_df

        concentration_col = self._first_existing_column(
            meta,
            [
                "Substrate_mM_added",
                "Substrate_mM",
                "Substrate_mmol",
                "Reference_mM",
                "Concentration_mM",
            ],
        )

        if concentration_col is None:
            raise KeyError(
                "No concentration column found in metadata. Expected one of: "
                "Substrate_mM_added, Substrate_mM, Substrate_mmol, Reference_mM, Concentration_mM. "
                f"Available columns: {meta.columns.tolist()}"
            )

        raw_value = str(meta.iloc[0][concentration_col])
        mmol_match = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw_value)

        if not mmol_match:
            raise ValueError(
                f"Could not extract numeric mmol value from metadata column "
                f"'{concentration_col}' with value '{raw_value}'."
            )

        mmol = float(mmol_match[0])
        water_amp_col = self._water_amp_column()
        water_amp_mean = self.fitting_params[water_amp_col].mean()

        if water_amp_mean == 0 or pd.isna(water_amp_mean):
            raise ValueError(f"Invalid water amplitude mean from column '{water_amp_col}': {water_amp_mean}")

        return mmol / water_amp_mean

    def plot(self, i):
        """
        Generate the original reference plot:
        left = water integral over time,
        right = reference spectrum with Lorentzian fit.
        """
        if self.data.shape[1] < 2:
            raise ValueError("Reference CSV must contain chemical shift column and at least one spectrum column.")

        # Clamp frame index safely. Existing UI uses 1-based frame numbers.
        max_frame = self.data.shape[1] - 1
        i = int(i)
        i = min(max(i, 1), max_frame)
        fit_idx = i if i in self.fitting_params.index else self.fitting_params.index[min(i - 1, len(self.fitting_params.index) - 1)]

        spectra_data = self.data.iloc[:, i]

        water_pos_col, water_width_col, water_amp_col = self._water_pos_width_amp_columns()

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        water_integral = self.fitting_params[water_amp_col] * np.pi
        ax[0].plot(water_integral)
        ax[0].axhline(y=water_integral.mean(), color="grey", linestyle="--")
        ax[0].set_title("Integral of water over time")
        ax[0].set_xlabel("Time step")
        ax[0].set_ylabel("Integral value water peak")
        ax[0].annotate(
            f"Calculated Conversion Factor = {self.reference_value / np.pi:.4f}",
            xy=(1.05, 0.85),
            xycoords="axes fraction",
            xytext=(-20, 20),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            va="top",
        )

        ax[1].plot(self.chem_shifts, spectra_data, c="blue", label="Reference spectrum")

        y_lorentzian = self.LorentzianFit.lorentzian(
            x=self.data.iloc[:, 0],
            shift=self.fitting_params.loc[fit_idx, water_pos_col],
            gamma=self.fitting_params.loc[fit_idx, water_width_col],
            A=self.fitting_params.loc[fit_idx, water_amp_col],
        )

        y_shift = self.fitting_params.loc[fit_idx, "y_shift"] if "y_shift" in self.fitting_params.columns else 0
        ax[1].plot(
            self.chem_shifts,
            y_lorentzian + y_shift,
            c="red",
            label="Lorentzian fit",
        )

        ax[1].set_xlabel("Chemical shift [ppm]")
        ax[1].set_ylabel("Intensity")
        ax[1].set_title(f"Lorentzian fit for time step: {i}")
        ax[1].set_xlim(max(self.chem_shifts), min(self.chem_shifts))
        ax[1].legend()

        fig.suptitle("Reference spectrum and Lorentzian fit of File: " + self.file_name_ref)
        plt.tight_layout()

        return fig

    def save_fig(self, fig, name, width=1200, height=800):
        """
        Save generated plot as PDF and PNG.
        """
        fig.set_size_inches(width / 100, height / 100)
        fig.savefig(f"{name}.pdf", format="pdf")
        fig.savefig(f"{name}.png", format="png")

    def save_kinetics_mmol(self):
        """
        Save kinetics converted to mmol using the reference value.

        Important:
        The old version expected fixed columns ['ReacSubs', 'Metab1', 'Water'].
        The current pipeline writes real labels such as Substrate, Water,
        Ethnaol-2-13C, Trehalose-1-13C, etc. Therefore we convert all numeric
        columns except Time_Step.
        """
        if "Time_Step" in self.kin_df.columns:
            kin_mmol = self.kin_df.copy().set_index("Time_Step")
        else:
            kin_mmol = self.kin_df.copy()

        numeric_cols = kin_mmol.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            raise ValueError(
                "No numeric kinetics columns found to convert. "
                f"Available columns: {kin_mmol.columns.tolist()}"
            )

        for col in numeric_cols:
            kin_mmol[col] = kin_mmol[col] * self.reference_value

        kin_mmol.to_csv(Path(self.output_dir, "kinetics_mmol.csv"))

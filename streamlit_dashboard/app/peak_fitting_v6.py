import pandas as pd
import os
import re
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
import streamlit as st

from peak_chunking import apply_metadata_chunking, label_from_metadata_column, log_chunking_stats


class PeakFitting:
    def __init__(
        self,
        fp_file,
        fp_meta,
        enable_chunking: bool = True,
        chunk_half_width: float = 0.5,
    ):
        self.fp_file = fp_file
        self.fp_meta = fp_meta
        self.file_name = os.path.splitext(os.path.basename(fp_file))[0]
        self.meta_name = os.path.basename(fp_meta)

        self.output_direc = os.path.join("output", self.file_name + "_output")
        os.makedirs(self.output_direc, exist_ok=True)

        # Read CSV robustly: supports comma/semicolon delimiters and malformed instrument rows.
        self.df = pd.read_csv(fp_file, sep=None, engine="python", on_bad_lines="skip")

        # Convert all spectrum values to numeric safely.
        for col in self.df.columns:
            if self.df[col].dtype == object:
                self.df[col] = self.df[col].astype(str).str.replace(",", ".", regex=False)
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        before_shape = self.df.shape
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=0, how="any", inplace=True)
        after_shape = self.df.shape
        if before_shape != after_shape:
            print(f"[PeakFitting] Cleaned data: {before_shape} -> {after_shape} (rows x cols)")

        self.meta_df = pd.read_excel(fp_meta)

        print("Loaded metadata columns:", self.meta_df.columns.tolist())

        metadata_file_col = self._get_first_existing_column(
            self.meta_df,
            ["File_Name_for_CSV", "File", "File_Name", "Filename", "Spectrum_File", "CSV_File"]
        )

        if metadata_file_col:
            print(
                f"Loaded metadata entries from '{metadata_file_col}':",
                self.meta_df[metadata_file_col].astype(str).tolist()
            )
        else:
            print(
                "No file-name column found in metadata. Available columns:",
                self.meta_df.columns.tolist()
            )

        self.number_time_points = self.df.shape[1] - 1
        self.time_points = np.arange(1, self.number_time_points + 1)
        self.x = self.df.iloc[:, 0]

        self.positions, self.names = self.extract_ppm_all()
        self.number_peaks = len(self.positions)
        self.number_substances = len(set(self.names))

        self.enable_chunking = enable_chunking
        self.chunk_half_width = chunk_half_width
        self._setup_chunking()

        column_names = (
            ["Time_Step"]
            + ["y_shift"]
            + [f"{name}_pos_{pos}" for name, pos in zip(self.names, self.positions)]
            + [f"{name}_width_{pos}" for name, pos in zip(self.names, self.positions)]
            + [f"{name}_amp_{pos}" for name, pos in zip(self.names, self.positions)]
        )

        self.fitting_params = pd.DataFrame(columns=column_names)
        self.fitting_params_error = pd.DataFrame(columns=column_names)

        self.fitting_params["Time_Step"] = self.time_points
        self.fitting_params_error["Time_Step"] = self.time_points

        self.fitting_params = self.fitting_params.set_index("Time_Step")
        self.fitting_params_error = self.fitting_params_error.set_index("Time_Step")

        self.names_substances = (
            deepcopy(self.names)
            + list(dict.fromkeys(self.names))
            + list(dict.fromkeys(self.names))
        )

    @staticmethod
    def _get_first_existing_column(df, possible_columns):
        for col in possible_columns:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _normalize_file_name(value):
        text = str(value).strip().lower()
        text = re.sub(r"\s+_", "_", text)
        return text

    @staticmethod
    def _normalize_experiment_id(value):
        text = str(value).strip().lower()
        return re.sub(r"\.0+$", "", text)

    def extract_ppm_all(self):
        filename_base = self._normalize_file_name(self.file_name)

        if filename_base.endswith("_b"):
            name = filename_base[:-2]
            parts = name.rsplit("_", 1)

            if len(parts) != 2:
                st.warning(
                    f"Could not split Bruker-converted file name '{self.file_name}' "
                    "into project name and experiment ID. Only water peak will be fitted."
                )
                return [4.7], ["Water"]

            required_cols = ["Project_Name", "Experiment_ID"]
            missing_cols = [col for col in required_cols if col not in self.meta_df.columns]
            if missing_cols:
                st.warning(
                    f"Missing metadata column(s) for Bruker matching: {missing_cols}. "
                    "Only water peak will be fitted."
                )
                return [4.7], ["Water"]

            project_name = self._normalize_file_name(parts[0])
            experiment_id = self._normalize_experiment_id(parts[1])

            filtered = self.meta_df[
                (self.meta_df["Project_Name"].astype(str).map(self._normalize_file_name) == project_name)
                & (self.meta_df["Experiment_ID"].astype(str).map(self._normalize_experiment_id) == experiment_id)
            ]

        else:
            file_col = self._get_first_existing_column(
                self.meta_df,
                ["File_Name_for_CSV", "File", "File_Name", "Filename", "Spectrum_File", "CSV_File"]
            )

            if file_col is None:
                st.warning(
                    "No metadata file-name column found. Expected one of: "
                    "File_Name_for_CSV, File, File_Name, Filename, Spectrum_File, CSV_File. "
                    "Only water peak will be fitted."
                )
                return [4.7], ["Water"]

            self.meta_df[file_col] = self.meta_df[file_col].astype(str).map(self._normalize_file_name)
            filtered = self.meta_df[self.meta_df[file_col].str.contains(filename_base, na=False)]

        if filtered.empty:
            st.warning(f"No metadata found for file '{self.file_name}'. Only water peak will be fitted.")
            return [4.7], ["Water"]

        self.meta_df = filtered.reset_index(drop=True)

        positions = []
        names = []

        substrate_name = label_from_metadata_column(self.meta_df, "Substrate", "Substrate")

        try:
            react_substrat = str(self.meta_df["Substrate_ppm"].iloc[0]).split(",")
            if react_substrat != ["nan"]:
                for val in react_substrat:
                    val = str(val).strip()
                    if val:
                        names.append(substrate_name)
                        positions.append(float(val))
        except Exception as e:
            print("No substrate found:", e)

        for i in range(1, 6):
            try:
                react_metabolite = str(self.meta_df[f"Metabolite_{i}_ppm"].iloc[0]).split(",")
                if react_metabolite != ["nan"]:
                    metabolite_name = label_from_metadata_column(
                        self.meta_df,
                        f"Metabolite_{i}",
                        f"Metabolite {i}"
                    )
                    for val in react_metabolite:
                        val = str(val).strip()
                        if val:
                            names.append(metabolite_name)
                            positions.append(float(val))
            except Exception:
                continue

        try:
            positions.append(float(self.meta_df["Water_ppm"].iloc[0]))
            names.append(label_from_metadata_column(self.meta_df, "Water", "Water"))
        except Exception:
            positions.append(4.7)
            names.append("Water")

        return positions, names

    def _setup_chunking(self):
        if not self.enable_chunking:
            self.fit_mask = np.ones(len(self.x), dtype=bool)
            self.chunk_stats = {}
            self.merged_chunks = []
            self.chunk_result = None
            return

        chunk_result = apply_metadata_chunking(
            self.df,
            metadata_peaks=self.positions,
            metadata_peak_names=self.names,
            half_width=self.chunk_half_width,
        )
        self.chunk_result = chunk_result
        self.fit_mask = chunk_result.mask
        self.merged_chunks = chunk_result.merged_chunks
        self.chunk_stats = chunk_result.stats
        log_chunking_stats(chunk_result)

    def _fitting_xy(self, y):
        if self.enable_chunking and hasattr(self, "fit_mask"):
            mask = self.fit_mask
            return self.x[mask].to_numpy(), y[mask].to_numpy()
        return self.x.to_numpy(), y.to_numpy()

    def make_bounds(
        self,
        mode,
        positions_fine=None,
        y_shift=(0, np.inf),
        shift_bounds=(-np.inf, np.inf),
        width_bounds=(0, 3e-1),
        amplitude_bounds=(0, np.inf),
        shift_bounds_fine=(-0.1, 0.1),
        width_bounds_fine=(0, 3e-1),
        amplitude_bounds_fine=(0, np.inf),
    ):
        if mode == "first":
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds = np.full(1, shift_bounds[0])
            shift_upper_bounds = np.full(1, shift_bounds[1])
            width_lower_bounds = np.full(self.number_substances, width_bounds[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds[1])
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds[0])
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds[1])

            return (
                np.concatenate([y_shift_lower_bounds, shift_lower_bounds, width_lower_bounds, amplitude_lower_bounds]),
                np.concatenate([y_shift_upper_bounds, shift_upper_bounds, width_upper_bounds, amplitude_upper_bounds]),
            )

        elif mode == "fine":
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds_fine = positions_fine + shift_bounds_fine[0]
            shift_upper_bounds_fine = positions_fine + shift_bounds_fine[1]
            width_lower_bounds = np.full(self.number_substances, width_bounds_fine[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds_fine[1])
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds_fine[0])
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds_fine[1])

            return (
                np.concatenate([y_shift_lower_bounds, shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds]),
                np.concatenate([y_shift_upper_bounds, shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds]),
            )

        raise ValueError(f"Unknown bounds mode: {mode}")

    def unpack_params_errors(self, popt, pcov):
        error = np.sqrt(np.diag(pcov))

        widths_final = []
        amplitudes_final = []
        widths_final_error = []
        amplitudes_final_error = []

        k = 0
        dummy = self.names[k]

        for name in self.names:
            if name != dummy:
                k += 1
                dummy = name

            widths_final_error.append(error[self.number_peaks + k + 1])
            amplitudes_final_error.append(error[self.number_peaks + self.number_substances + k + 1])
            widths_final.append(popt[self.number_peaks + k + 1])
            amplitudes_final.append(popt[self.number_peaks + self.number_substances + k + 1])

        return (
            np.concatenate([np.array([popt[0]]), popt[1:self.number_peaks + 1], widths_final, amplitudes_final]),
            np.concatenate([np.array([error[0]]), error[1:self.number_peaks + 1], widths_final_error, amplitudes_final_error]),
        )

    def fit(self, save_csv=True):
        flattened_bounds = self.make_bounds(mode="first")

        progress_bar = st.empty()
        load_bar = progress_bar.progress(0)
        first_fit = True

        for i in tqdm(range(self.number_time_points), desc=self.file_name):
            try:
                y = self.df.iloc[:, i + 1]
                x_fit, y_fit = self._fitting_xy(y)

                if first_fit:
                    popt, pcov = curve_fit(
                        lambda x, *params: self.grey_spectrum(x, *params),
                        x_fit,
                        y_fit,
                        p0=[0] + [0] + [0.1] * self.number_substances + [1000] * self.number_substances,
                        maxfev=3000,
                        ftol=1e-1,
                        xtol=1e-1,
                        bounds=flattened_bounds,
                    )

                    y_shift = np.array([popt[0]])
                    positions_fine = popt[1] + np.array(self.positions)
                    widths = popt[2:self.number_substances + 2]
                    amplitudes = popt[self.number_substances + 2:]

                    p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])
                    flattened_bounds_fine = self.make_bounds(mode="fine", positions_fine=positions_fine)
                    first_fit = False

                popt, pcov = curve_fit(
                    lambda x, *params: self.grey_spectrum_fine_tune(x, *params),
                    x_fit,
                    y_fit,
                    p0=p0,
                    maxfev=20000,
                    bounds=flattened_bounds_fine,
                    ftol=1e-6,
                    xtol=1e-6,
                )

                y_shift = np.array([popt[0]])
                positions_fine = popt[1:self.number_peaks + 1]
                widths = popt[1 + self.number_peaks:self.number_peaks + self.number_substances + 1]
                amplitudes = popt[self.number_peaks + self.number_substances + 1:]

                p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])

                self.fitting_params.loc[i + 1], self.fitting_params_error.loc[i + 1] = self.unpack_params_errors(popt, pcov)

            except RuntimeError:
                print(f"Could not fit time frame number {i}. Skipping...")

            except Exception as e:
                print(f"Unexpected fitting error at time frame {i}: {e}. Skipping...")

            load_bar.progress((i + 1) / self.number_time_points)

        progress_bar.empty()

        self.fitting_params.fillna(0, inplace=True)
        self.fitting_params_error.fillna(0, inplace=True)

        if save_csv:
            self.fitting_params.to_csv(os.path.join(self.output_direc, "fitting_params.csv"))
            self.fitting_params_error.to_csv(os.path.join(self.output_direc, "fitting_params_error.csv"))
        else:
            return self.fitting_params

    def lorentzian(self, x, shift, gamma, A):
        return A * gamma / ((x - shift) ** 2 + gamma ** 2)

    def grey_spectrum(self, x, *params):
        y_shift = params[0]
        shift = params[1]
        gamma = params[2:self.number_substances + 2]
        A = params[self.number_substances + 2:]

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0]

        for i in range(self.number_peaks):
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]

            if k < self.number_substances:
                y += self.lorentzian(x, shift + self.positions[i], gamma[k], A[k]) + y_shift

        return y

    def write_results(self):
        self.fitting_params.to_csv(os.path.join(self.output_direc, "fitting_params.csv"))
        self.fitting_params_error.to_csv(os.path.join(self.output_direc, "fitting_params_error.csv"))

    def grey_spectrum_fine_tune(self, x, *params):
        y_shift = params[0]
        x0 = params[1:self.number_peaks + 1]
        gamma = params[1 + self.number_peaks:self.number_peaks + self.number_substances + 1]
        A = params[self.number_peaks + self.number_substances + 1:]

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0]

        for i in range(self.number_peaks):
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]

            if k < self.number_substances:
                y += self.lorentzian(x, x0[i], gamma[k], A[k]) + y_shift

        return y

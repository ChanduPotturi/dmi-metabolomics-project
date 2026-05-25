import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

import io
import nmrglue as ng
import numpy as np
from scipy.signal import find_peaks
import sys
import ctypes
import tkinter as tk
from tkinter import TclError
from tkinter import filedialog

from LoadData import *
from Process4Panels import Process4Panels
from panel1_spectrum_plot import Panel1SpectrumPlot
from panel2_kinetic_plot import KineticPlot
from panel3_contour_plot import ContourPlot
from panel5_reference_plot import Reference
from panel6_stacked_plot import StackedSpectraPlot


st.set_page_config(layout="wide", page_title="SBMI - Application", page_icon=":shark:")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def process_to_dataframe(pdata_path: Path) -> pd.DataFrame:
    """Read Bruker pdata/1 and return the cropped DataFrame (ppm + active cols)."""
    dic, data = ng.bruker.read_pdata(str(pdata_path))
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic, dim=1)
    ppm = uc.ppm_scale()

    df = pd.DataFrame(data.T)
    df.insert(0, "ppm", ppm)
    df = df.sort_values("ppm", ascending=True)

    # Keep only active spectra columns and match the notebook downsampling behavior.
    spectra = df.iloc[:, 1:]
    mask = spectra.abs().max(axis=0) > 1e6
    df = pd.concat([df["ppm"], spectra.loc[:, mask]], axis=1)
    df = df.iloc[1::2, :].reset_index(drop=True)

    scale_factor = 32000
    df.loc[:, df.columns != "ppm"] = df.loc[:, df.columns != "ppm"] / scale_factor

    # Rename ppm column to match the expected name in the notebook
    df.rename(columns={df.columns[0]: "2H chemical shift (ppm)"}, inplace=True)

    # Align first frame to 4.7 ppm.
    ppm_vals = df.iloc[:, 0].values
    first = df.iloc[:, 1].to_numpy()
    peaks, _ = find_peaks(first, prominence=np.std(first))
    if len(peaks) == 0:
        peaks, _ = find_peaks(first)
    if len(peaks) == 0:
        raise ValueError("No peaks detected for alignment")
    closest = peaks[np.argmin(np.abs(ppm_vals[peaks] - 4.7))]
    df["2H chemical shift (ppm)"] = df["2H chemical shift (ppm)"] + (4.7 - ppm_vals[closest])

    # Crop to the detected peak region from the mean spectrum.
    area_around = 0.5
    ppm_vals = df.iloc[:, 0].values
    mean_spec = df.iloc[:, 1:].mean(axis=1).values
    peaks, _ = find_peaks(
        mean_spec,
        prominence=np.std(mean_spec) * 2,
        height=np.mean(mean_spec) + 1.5 * np.std(mean_spec),
    )
    if len(peaks) == 0:
        raise ValueError("No peaks detected in mean spectrum")
    pppm = ppm_vals[peaks]
    low, high = pppm.min() - area_around, pppm.max() + area_around
    mask_range = (ppm_vals >= low) & (ppm_vals <= high)
    return df.loc[mask_range].reset_index(drop=True)


def file_signature(path):
    if path and os.path.exists(path):
        return os.path.getmtime(path)
    return None


def output_folder_signature(folder_path):
    if not os.path.exists(folder_path):
        return None

    sig = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            p = os.path.join(root, file)
            try:
                sig.append((p, os.path.getmtime(p), os.path.getsize(p)))
            except OSError:
                pass
    return tuple(sig)


@st.cache_data(show_spinner=False)
def create_zip_bytes_cached(folder_path, folder_signature):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zip_file.write(full_path, arcname)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@st.cache_data(show_spinner=True)
def run_processing_cached(data_path, meta_path, ref_path, model_name, data_sig, meta_sig, ref_sig):
    if model_name == "Model 1":
        from peak_fitting_v6 import PeakFitting
    else:
        from peak_fitting_v7 import PeakFitting

    fitter = PeakFitting(data_path, meta_path)
    fitter.fit()

    fitting_params_path = os.path.join(fitter.output_direc, "fitting_params.csv")
    fitting_params_error_path = os.path.join(fitter.output_direc, "fitting_params_error.csv")

    fitter.fitting_params.to_csv(fitting_params_path)
    fitter.fitting_params_error.to_csv(fitting_params_error_path)

    processor = Process4Panels(data_path)
    processor.save_sum_spectra()
    processor.save_substrate_individual()
    processor.save_individual_peaks()
    processor.save_difference()
    processor.save_kinetics()

    if ref_path and os.path.exists(ref_path):
        try:
            ref_processor = Reference(
                fp_ref=ref_path,
                fp_meta=meta_path,
                fp_data=data_path
            )
            ref_processor.save_kinetics_mmol()
        except Exception:
            pass

    return {
        "data_path": data_path,
        "meta_path": meta_path,
        "ref_path": ref_path,
        "file_name": os.path.splitext(os.path.basename(data_path))[0],
    }


@st.cache_resource(show_spinner=False)
def get_panel1_cached(data_path, sig):
    return Panel1SpectrumPlot(file_path=data_path)


@st.cache_resource(show_spinner=False)
def get_panel2_cached(data_path, sig):
    return KineticPlot(path=data_path)


@st.cache_resource(show_spinner=False)
def get_panel3_cached(data_path, sig):
    return ContourPlot(file_path=data_path)


@st.cache_resource(show_spinner=False)
def get_panel6_cached(data_path, sig):
    return StackedSpectraPlot(file_path=data_path)


@st.cache_resource(show_spinner=False)
def get_reference_cached(ref_path, meta_path, data_path, ref_sig, meta_sig, data_sig):
    obj = Reference(fp_ref=ref_path, fp_meta=meta_path, fp_data=data_path)
    obj.save_kinetics_mmol()
    return obj


class StreamlitApp:
    def __init__(self):
        pass

    def _get_upload_dir(self):
        upload_dir = Path("runtime_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir

    def _save_uploaded_file(self, uploaded_file):
        upload_dir = self._get_upload_dir()
        target_path = upload_dir / uploaded_file.name

        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(target_path.resolve())

    def safe_frame_slider(self, label, n_frames, default=1):
        if n_frames <= 1:
            st.info("Only one frame available.")
            return 1

        return st.slider(
            label,
            min_value=1,
            max_value=n_frames,
            value=min(default, n_frames)
        )

    def save_matplotlib_formats(
        self,
        session_obj=None,
        fig=None,
        file_basename=None,
        file_name=None,
        button_key=None
    ):
        if fig is None:
            st.error("No figure available for export.")
            return

        if file_basename is None:
            file_basename = "plot"

        if file_name is None:
            file_name = "plot"

        if button_key is None:
            button_key = file_name

        plot_dir = Path("output", file_basename + "_output", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        st.markdown("**Select export format(s)**")
        c1, c2, c3 = st.columns(3)

        with c1:
            save_pdf = st.checkbox("PDF", key=f"pdf_{button_key}")
        with c2:
            save_png = st.checkbox("PNG", key=f"png_{button_key}")
        with c3:
            save_jpg = st.checkbox("JPG", key=f"jpg_{button_key}")

        if st.button("Save Selected Format(s)", key=f"save_btn_{button_key}"):
            if not (save_pdf or save_png or save_jpg):
                st.warning("Please select at least one format.")
                return

            saved_files = []

            try:
                if save_pdf:
                    pdf_path = Path(plot_dir, f"{file_name}_{file_basename}.pdf")
                    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
                    saved_files.append(str(pdf_path))

                if save_png:
                    png_path = Path(plot_dir, f"{file_name}_{file_basename}.png")
                    fig.savefig(str(png_path), format="png", dpi=180, bbox_inches="tight")
                    saved_files.append(str(png_path))

                if save_jpg:
                    jpg_path = Path(plot_dir, f"{file_name}_{file_basename}.jpg")
                    fig.savefig(str(jpg_path), format="jpeg", dpi=180, bbox_inches="tight")
                    saved_files.append(str(jpg_path))

                st.session_state[f"saved_files_{button_key}"] = saved_files
                st.success("Selected format(s) saved successfully.")

            except Exception as e:
                st.error(f"Failed to save selected format(s): {e}")

        saved_files = st.session_state.get(f"saved_files_{button_key}", [])

        if saved_files:
            mime_map = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg"
            }

            for idx, saved_file in enumerate(saved_files):
                if os.path.exists(saved_file):
                    with open(saved_file, "rb") as f:
                        ext = Path(saved_file).suffix.lower()
                        st.download_button(
                            label=f"Download {Path(saved_file).name}",
                            data=f.read(),
                            file_name=os.path.basename(saved_file),
                            mime=mime_map.get(ext, "application/octet-stream"),
                            key=f"download_{button_key}_{idx}"
                        )

    def save_plotly_formats(self, fig, file_basename, file_name, button_key):
        plot_dir = Path("output", file_basename + "_output", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        st.markdown("**Select export format(s)**")
        c1, c2, c3 = st.columns(3)

        with c1:
            save_pdf = st.checkbox("PDF", key=f"plotly_pdf_{button_key}")
        with c2:
            save_png = st.checkbox("PNG", key=f"plotly_png_{button_key}")
        with c3:
            save_jpg = st.checkbox("JPG", key=f"plotly_jpg_{button_key}")

        if st.button("Save Selected Format(s)", key=f"plotly_save_btn_{button_key}"):
            if not (save_pdf or save_png or save_jpg):
                st.warning("Please select at least one format.")
                return

            saved_files = []

            try:
                if save_pdf:
                    pdf_path = Path(plot_dir, f"{file_name}_{file_basename}.pdf")
                    pio.write_image(fig, str(pdf_path), format="pdf", engine="kaleido", width=1000, height=650)
                    saved_files.append(str(pdf_path))

                if save_png:
                    png_path = Path(plot_dir, f"{file_name}_{file_basename}.png")
                    pio.write_image(fig, str(png_path), format="png", engine="kaleido", width=1000, height=650)
                    saved_files.append(str(png_path))

                if save_jpg:
                    jpg_path = Path(plot_dir, f"{file_name}_{file_basename}.jpg")
                    pio.write_image(fig, str(jpg_path), format="jpg", engine="kaleido", width=1000, height=650)
                    saved_files.append(str(jpg_path))

                st.session_state[f"plotly_saved_files_{button_key}"] = saved_files
                st.success("Selected format(s) saved successfully.")

            except Exception as e:
                st.error(f"Failed to export Plotly figure: {e}")

        saved_files = st.session_state.get(f"plotly_saved_files_{button_key}", [])

        if saved_files:
            mime_map = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg"
            }

            for idx, saved_file in enumerate(saved_files):
                if os.path.exists(saved_file):
                    with open(saved_file, "rb") as f:
                        ext = Path(saved_file).suffix.lower()
                        st.download_button(
                            label=f"Download {Path(saved_file).name}",
                            data=f.read(),
                            file_name=os.path.basename(saved_file),
                            mime=mime_map.get(ext, "application/octet-stream"),
                            key=f"plotly_download_{button_key}_{idx}"
                        )

    def _pick_folder(self, title):
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(1)
                except Exception:
                    try:
                        ctypes.windll.user32.SetProcessDPIAware()
                    except Exception:
                        pass

        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title=title)
            return Path(folder) if folder else None
        except (TclError, RuntimeError, OSError) as exc:
            st.warning(f"Could not open system folder picker. You can paste the folder path manually. Details: {exc}")
            return None
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass

    def _find_bruker_pdata(self, selected_folder):
        selected_folder = Path(selected_folder)

        direct_candidates = [
            selected_folder,
            selected_folder / "pdata" / "1",
            selected_folder / "1",
        ]
        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_dir() and candidate.name == "1" and candidate.parent.name == "pdata":
                return candidate

        for candidate in selected_folder.rglob("1"):
            if candidate.is_dir() and candidate.parent.name == "pdata":
                return candidate

        return None


    def _prepare_bruker_input(self, selected_folder):
        pdata_path = self._find_bruker_pdata(selected_folder)
        if pdata_path is None:
            raise ValueError("Could not find a Bruker pdata/1 folder in the selected directory.")

        bruker_df = process_to_dataframe(pdata_path)
        csv_buffer = io.BytesIO(bruker_df.to_csv(index=False).encode("utf-8"))
        run_name = pdata_path.parent.parent.name if pdata_path.parent.parent.name else Path(selected_folder).name
        sample_name = pdata_path.parent.parent.parent.name if pdata_path.parent.parent.parent.name else Path(selected_folder).name
        source_name = f"{sample_name}_{run_name}"
        csv_buffer.name = f"{source_name}.csv"
        return csv_buffer, pdata_path

    def header(self):
        st.markdown("""<h1 style="text-align: center;">SBMI - Application</h1>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([0.05, 0.9, 0.05])

        with col1:
            st.divider()

        with col2:
            options = ["Model 1: Lorentzian Fit", "Model 2: Lorentzian fit with prefitting"]
            selected_option = st.selectbox("Choose the Model:", options)

            if selected_option == "Model 1: Lorentzian Fit":
                st.session_state["Model 1"] = True
                st.session_state["Model 2"] = False
                model_name = "Model 1"
            else:
                st.session_state["Model 1"] = False
                st.session_state["Model 2"] = True
                model_name = "Model 2"

            st.session_state["model_name"] = model_name

            input_options = ["CSV", "Bruker"]
            input_source = st.selectbox("Choose input type:", input_options)
            st.session_state["input_source"] = input_source

            st.divider()

            if input_source == "CSV":
                batch_mode = st.checkbox(
                    "Batch Processing Mode",
                    value=False,
                    help="Process multiple CSV files at once"
                )
            else:
                batch_mode = False
                st.session_state["batch_mode"] = False
                st.info("Bruker input is converted to CSV and processed as a single dataset.")
            st.session_state["batch_mode"] = batch_mode

            sub_col1, sub_col2 = st.columns([0.30, 0.70])

            with sub_col1:
                st.markdown("**Step 1: Select the Metadata as .xlsx**")
                self.meta_fp = st.file_uploader("Step 1: Upload Metadata (.xlsx)", type=["xlsx"], key="meta_file")

# Select the Spectrum as csv 
                if input_source == "CSV":
                    if not batch_mode:
                        st.markdown('**Step 2: Select Spectrum as .csv**')
                        self.data_fp = st.file_uploader("Step 2: Upload Spectrum (.csv)", type=["csv"], key="spectrum_file")
                        self.data_files = [self.data_fp] if self.data_fp else []

                    else:
                    # BATCH MODE - Multiple Files
                        st.markdown('**Step 2: Select Multiple Spectra as .csv**')
                        uploaded_files = st.file_uploader(
                            "Upload multiple Spectrum files (.csv)",
                            type=["csv"],
                            accept_multiple_files=True,
                            key="spectrum_files_batch"
                        )
                        self.data_files = uploaded_files if uploaded_files else []
                        self.data_fp = None  
                        
                        if self.data_files:
                            st.info(f"📁 {len(self.data_files)} files selected")
                else:
                    st.markdown('**Step 2: Select a Bruker folder**')
                    if st.button("Choose Bruker folder", key="choose_bruker_folder"):
                        selected_folder = self._pick_folder("Select the Bruker pdata/1 folder")
                        if selected_folder is not None:
                            try:
                                bruker_buffer, bruker_pdata_path = self._prepare_bruker_input(selected_folder)
                                st.session_state["bruker_selected_folder"] = str(selected_folder)
                                st.session_state["bruker_pdata_path"] = str(bruker_pdata_path)
                                st.session_state["bruker_csv_bytes"] = bruker_buffer.getvalue()
                                st.session_state["bruker_csv_name"] = bruker_buffer.name
                            except Exception as exc:
                                st.session_state.pop("bruker_csv_bytes", None)
                                st.session_state.pop("bruker_csv_name", None)
                                st.session_state.pop("bruker_selected_folder", None)
                                st.session_state.pop("bruker_pdata_path", None)
                                st.error(f"Bruker conversion failed: {exc}")

                    if st.session_state.get("bruker_csv_bytes"):
                        self.data_fp = io.BytesIO(st.session_state["bruker_csv_bytes"])
                        self.data_fp.name = st.session_state.get("bruker_csv_name", "bruker.csv")
                        self.data_files = [self.data_fp]
                        st.success(f"Bruker folder selected: {st.session_state.get('bruker_selected_folder', '')}")
                        st.info(f"Converted file: {self.data_fp.name}")
                    else:
                        self.data_fp = None
                        self.data_files = []
                        st.warning("No Bruker folder selected yet.")

                apply_reference = st.checkbox(
                    "Apply signal referencing (optional)",
                    value=True,
                    help="If unchecked, no reference file is used and results stay in arbitrary units (a.u.) instead of mmol."
                )
                st.session_state["use_reference"] = apply_reference

                if st.session_state.get("use_reference", True):
                    st.markdown("**Step 3: Select the Reference File as .csv**")
                    self.reference_fp = st.file_uploader("Step 3: Upload Reference (.csv)", type=["csv"], key="reference_file")
                else:
                    self.reference_fp = None
                    st.info("Reference file disabled. Results will be in arbitrary units (a.u.).")

        with sub_col2:
            st.markdown("**Uploaded Files**")

            if self.meta_fp is not None:
                st.success(f"Metadata file uploaded: {self.meta_fp.name}")
            else:
                st.warning("No metadata file uploaded.")

            if batch_mode:
                if self.data_files:
                    st.success(f"{len(self.data_files)} spectrum files uploaded:")
                    with st.expander("View file list"):
                        for i, file in enumerate(self.data_files, 1):
                            st.text(f"{i}. {file.name}")
                else:
                    st.warning("No spectrum files uploaded.")
            else:
                if self.data_fp:
                    st.success(f"Spectrum: {self.data_fp.name}")
                else:
                    st.warning("No spectrum file uploaded.")

            if self.reference_fp is not None:
                st.success(f"Reference file uploaded: {self.reference_fp.name}")
            else:
                st.warning("No reference file uploaded.")

        with col2:
            process_col1, process_col2, process_col3 = st.columns([2, 1, 1])

        with process_col1:
            if batch_mode:
                st.markdown("**Step 4: Press Start Batch Processing**")
            else:
                st.markdown("**Step 4: Press Start Processing**")

        with process_col2:
            can_process = self.meta_fp and len(self.data_files) > 0

            if st.button("Start Processing", disabled=not can_process):
                st.session_state["button_pressed"] = True
                st.session_state["processing_started"] = True

        if st.session_state.get("button_pressed", False):
            st.session_state["button_pressed"] = False

            if batch_mode:
                self.process_batch()
                if st.session_state.get("batch_results", {}).get("successful"):
                    self.process_plots()
            else:
                with st.spinner("Processing data. First run may take time; repeated runs use cache..."):
                    self.process_data()
                    self.process_plots()

        with col3:
            st.divider()

        main, about = st.tabs(["Main Page", "Instructions"])

        if st.session_state.get("processing_started", False):
            if (
                st.session_state.get("file_name") is not None
                or (st.session_state.get("batch_mode") and st.session_state.get("batch_results"))
            ):
                self.main_page(main)

        self.about_page(about)

    def main_page(self, main):
        with main:
            st.markdown("#### Main Page Content")

            if st.session_state.get("batch_mode", False):
                if st.session_state.get("batch_results"):
                    results = st.session_state["batch_results"]
                    st.success(
                        f"Batch processing completed: {len(results['successful'])} of {results['total']} files processed successfully"
                    )
                else:
                    st.info("Click 'Start Processing' to process multiple files.")

            if st.session_state.get("processing_started", False):
                st.info(
                    "For faster interaction, each panel renders only when enabled. "
                    "Plot controls update only after clicking Apply."
                )

                show_panel1 = st.checkbox(
                    "Show Panel 1 - Substrate Plot",
                    value=True,
                    key="show_panel1"
                )

                show_panel2 = st.checkbox(
                    "Show Panel 2 - Kinetic Plot",
                    value=True,
                    key="show_panel2"
                )

                show_panel3 = st.checkbox(
                    "Show Panel 3 - Contour Plot",
                    value=False,
                    key="show_panel3"
                )

                show_panel4 = st.checkbox(
                    "Show Panel 4 - Reference Plot",
                    value=False,
                    key="show_panel4"
                )

                show_panel6 = st.checkbox(
                    "Show Panel 6 - Waterfall Plot",
                    value=False,
                    key="show_panel6"
                )

                if show_panel1:
                    self.panel1()

                if show_panel2:
                    self.panel2()

                if show_panel3:
                    self.panel3()

                if show_panel4:
                    self.panel4()

                if show_panel6:
                    self.panel6()

                st.markdown("---")
                st.markdown("### Download Results")

                output_dir = os.path.abspath("output")

                if os.path.exists(output_dir):
<<<<<<< HEAD
                    zip_path = os.path.join(tempfile.gettempdir(), "results.zip")
                    shutil.make_archive(zip_path.replace(".zip", ""), "zip", output_dir)

                    with open(zip_path, "rb") as f:
                        zip_bytes = f.read()

                    st.download_button(
                        label="📥 Download All Results (ZIP)",
                        data=zip_bytes,
                        file_name="results.zip",
                        mime="application/zip",
                        key="download_zip"
                    )
=======
                    try:
                        sig = output_folder_signature(output_dir)
                        zip_bytes = create_zip_bytes_cached(output_dir, sig)

                        st.download_button(
                            label="📥 Download All Results (ZIP)",
                            data=zip_bytes,
                            file_name="results.zip",
                            mime="application/zip",
                            key="download_zip"
                        )

                    except Exception as e:
                        st.error(f"ZIP creation failed: {e}")
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
                else:
                    st.warning("Output folder not found.")

            else:
                st.info("Click 'Start Processing' to see the analysis panels.")

    def process_data(self):
        if not self.data_fp or not self.meta_fp:
            st.error("Please upload both the spectrum (.csv) and metadata (.xlsx) files before processing.")
            return

        data_path = self._save_uploaded_file(self.data_fp)
        meta_path = self._save_uploaded_file(self.meta_fp)
        ref_path = self._save_uploaded_file(self.reference_fp) if self.reference_fp else None

        result = run_processing_cached(
            data_path=data_path,
            meta_path=meta_path,
            ref_path=ref_path,
            model_name=st.session_state.get("model_name", "Model 1"),
            data_sig=file_signature(data_path),
            meta_sig=file_signature(meta_path),
            ref_sig=file_signature(ref_path) if ref_path else None
        )

        st.session_state["tmp_data_path"] = result["data_path"]
        st.session_state["tmp_meta_path"] = result["meta_path"]
        st.session_state["tmp_ref_path"] = result["ref_path"]
        st.session_state["file_name"] = result["file_name"]

    def process_batch(self):
        if not self.meta_fp or not self.data_files:
            st.error("Please upload metadata and at least one spectrum file.")
            return

        meta_path = self._save_uploaded_file(self.meta_fp)
        ref_path = self._save_uploaded_file(self.reference_fp) if self.reference_fp else None

        total_files = len(self.data_files)
        progress_bar = st.progress(0)
        status_text = st.empty()

        successful_files = []
        failed_files = []

        for idx, data_file in enumerate(self.data_files):
            status_text.text(f"Processing file {idx + 1}/{total_files}: {data_file.name}")
            progress_bar.progress((idx + 1) / total_files)

            try:
                data_path = self._save_uploaded_file(data_file)

                run_processing_cached(
                    data_path=data_path,
                    meta_path=meta_path,
                    ref_path=ref_path,
                    model_name=st.session_state.get("model_name", "Model 1"),
                    data_sig=file_signature(data_path),
                    meta_sig=file_signature(meta_path),
                    ref_sig=file_signature(ref_path) if ref_path else None
                )

                successful_files.append(data_file.name)

            except Exception as e:
                failed_files.append((data_file.name, str(e)))
                st.error(f"Failed to process {data_file.name}: {str(e)}")

        progress_bar.progress(1.0)
        status_text.empty()

        st.success("Batch processing completed!")

        col_success, col_failed = st.columns(2)

        with col_success:
            st.metric("Successful", len(successful_files))
            if successful_files:
                with st.expander("View successful files"):
                    for file in successful_files:
                        st.text(file)

        with col_failed:
            st.metric("Failed", len(failed_files))
            if failed_files:
                with st.expander("View failed files"):
                    for file, error in failed_files:
                        st.text(f"Error: {file}")
                        st.caption(f"Error: {error}")

        st.session_state["batch_results"] = {
            "successful": successful_files,
            "failed": failed_files,
            "total": total_files
        }

        if successful_files:
            last_file = successful_files[-1]
            last_path = str((self._get_upload_dir() / last_file).resolve())
            st.session_state["tmp_data_path"] = last_path
            st.session_state["file_name"] = os.path.splitext(last_file)[0]
            st.session_state["tmp_meta_path"] = meta_path
            st.session_state["tmp_ref_path"] = ref_path

        st.session_state["processing_done"] = True

    def process_plots(self):
        data_path = st.session_state.get("tmp_data_path")
        meta_path = st.session_state.get("tmp_meta_path")
        ref_path = st.session_state.get("tmp_ref_path")
        use_reference = st.session_state.get("use_reference", True)

        if not data_path or not os.path.exists(data_path):
            st.error(f"Processed spectrum file not found: {data_path}")
            return

        data_sig = file_signature(data_path)

        st.session_state["panel_1_obj"] = get_panel1_cached(data_path, data_sig)
        st.session_state["panel_2_obj"] = get_panel2_cached(data_path, data_sig)
        st.session_state["panel_3_obj"] = get_panel3_cached(data_path, data_sig)
        st.session_state["panel_6_obj"] = get_panel6_cached(data_path, data_sig)

        if use_reference and ref_path and os.path.exists(ref_path):
            try:
                st.session_state["panel_4_obj"] = get_reference_cached(
                    ref_path,
                    meta_path,
                    data_path,
                    file_signature(ref_path),
                    file_signature(meta_path),
                    data_sig
                )
                st.success("Reference processing completed. Results available in mmol.")
            except Exception as e:
                st.error(f"Error processing reference: {str(e)}")
                st.session_state["panel_4_obj"] = None
        else:
            st.session_state["panel_4_obj"] = None
            if not use_reference:
                st.info("Reference processing skipped. Kinetic results remain in arbitrary units (a.u.).")
            else:
                st.warning("No reference file provided. Panel 4 will be skipped.")

<<<<<<< HEAD
        st.session_state["panel_6_obj"] = StackedSpectraPlot(file_path=tmp_data_path)

    def about_page(self, about):
        with about:
            st.markdown(
                """
                ### Instructions:

                #### Step 0:
                **Choose the Model:**

                **Model 1:**
                Lorenzian curve fitting with parameters of the Meta Description

                **Model 2:**
                Lorenzian curve fitting + initial parameters derived from actual spectrum

                #### Step 1:
                - **Select the Metadata**
                - **Select the Substrate**
                - **Select the Reference File**

                #### Step 2:
                - Click Start Processing

                #### Step 3:
                ##### Substrate Plot
                - Use sliders to select the frame
                - Use the legend to show or hide lines
                - Select export format before saving

                #### Kinetic Plot
                - Kinetics of metabolites and substrate peaks
                - Select export format before saving

                #### Contour Plot
                - Visualize the full measured spectrum and select the depth [%]
                - Select export format before saving

                #### Stacked Spectra Plot
                - Visualize multiple spectra stacked vertically
                - Adjust frame range and spacing
                - Select export format before saving

                #### Reference Plot
                - Get the reference value on water
                - Select export format before saving
                """
            )
        return None

    def safe_frame_slider(self, label, n_frames, default=1):
        if n_frames <= 1:
            st.info("Only one frame available.")
            return 1

        return st.slider(
            label,
            min_value=1,
            max_value=n_frames,
            value=min(default, n_frames)
        )

    def save_plot_with_format(self, session_obj, fig, file_basename, file_name, button_key, selected_format):
        plot_dir = Path("output", file_basename + "_output", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        ext = selected_format.lower()
        save_path = Path(plot_dir, f"{file_name}_{file_basename}.{ext}")

        if st.button(f"Save as {selected_format}", key=f"save_{button_key}_{ext}"):
            if ext == "pdf":
                session_obj.save_fig(fig=fig, name=save_path.with_suffix(""))
            elif ext in ["png", "jpg", "jpeg"]:
                if hasattr(fig, "write_image"):
                    plotly_format = "jpeg" if ext == "jpg" else ext
                    fig.write_image(
                        str(save_path),
                        format=plotly_format,
                        engine="kaleido",
                        width=1000,
                        height=700,
                        scale=1
                    )
                else:
                    matplotlib_format = "jpeg" if ext == "jpg" else ext
                    fig.savefig(str(save_path), format=matplotlib_format, dpi=300, bbox_inches="tight")
            else:
                st.error(f"Unsupported format: {selected_format}")
                return

            st.session_state[f"saved_file_{button_key}"] = str(save_path)
            st.success(f"Saved: {save_path}")

        saved_file = st.session_state.get(f"saved_file_{button_key}")
        if saved_file and os.path.exists(saved_file):
            mime_map = {
                "pdf": "application/pdf",
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg"
            }
            ext_saved = Path(saved_file).suffix.lower().replace(".", "")
            with open(saved_file, "rb") as f:
                st.download_button(
                    label=f"Download {Path(saved_file).name}",
                    data=f.read(),
                    file_name=os.path.basename(saved_file),
                    mime=mime_map.get(ext_saved, "application/octet-stream"),
                    key=f"download_{button_key}_{ext_saved}"
                )

=======
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
    def panel1(self):
        tmp_data_path = st.session_state.get("tmp_data_path")

        if not tmp_data_path or not os.path.exists(tmp_data_path):
            st.error("No processed data found. Please process data first.")
            return

        if "panel_1_obj" not in st.session_state or st.session_state["panel_1_obj"] is None:
            st.error("Panel 1 object not initialized. Please process data first.")
            return

        file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]

        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            panel1_obj = st.session_state["panel_1_obj"]
            data = panel1_obj.data
            diffs = panel1_obj.differences

            n_frames_data = max(data.shape[1] - 1, 0)

            if n_frames_data < 1:
                st.info("Panel 1: No time frames found.")
                return

            with st.form(f"panel1_form_{file_name}"):
                time_frame = st.slider(
                    "Select the frame",
                    min_value=1,
                    max_value=n_frames_data,
                    value=st.session_state.get(f"panel1_frame_{file_name}", 1),
                )
                apply_panel1 = st.form_submit_button("Apply Panel 1 Settings")

            if apply_panel1 or f"panel1_frame_{file_name}" not in st.session_state:
                st.session_state[f"panel1_frame_{file_name}"] = time_frame

            selected_frame = st.session_state.get(f"panel1_frame_{file_name}", 1)

            st.markdown("# Substrate Plot")

            noise_std_text = "n/a"
            if isinstance(diffs, pd.DataFrame) and diffs.shape[1] > selected_frame:
                noise_std = diffs.iloc[:, selected_frame].std()
                noise_std_text = f"{noise_std:.3f}"

            st.write(f"Standard deviation of noise: {noise_std_text}")

            one_plot = panel1_obj.plot(selected_frame)

            st.plotly_chart(
                one_plot,
                use_container_width=True,
                config={"displayModeBar": True}
            )

            self.save_plotly_formats(
                fig=one_plot,
                file_basename=file_name,
                file_name=f"Substrate_{file_name}_{selected_frame}",
                button_key=f"panel1_{file_name}_{selected_frame}"
            )

<<<<<<< HEAD
            self.save_plot_with_format(
                session_obj=st.session_state["panel_1_obj"],
                fig=one_plot,
                file_basename=file_name,
                file_name=f"Substrate_{file_name}_{time_frame}",
                button_key=f"panel1_{file_name}_{time_frame}",
                selected_format=export_format
            )

=======
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
    def panel2(self):
        tmp_data_path = st.session_state.get("tmp_data_path")

        if "panel_2_obj" not in st.session_state or st.session_state["panel_2_obj"] is None:
            st.error("Panel 2 object not initialized. Please process data first.")
            return

        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown("# Kinetic Plot")

            fig = st.session_state["panel_2_obj"].plot()
            file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True}
            )

            self.save_plotly_formats(
                fig=fig,
                file_basename=file_name,
                file_name=f"Kinetic_{file_name}",
                button_key=f"panel2_{file_name}"
            )

<<<<<<< HEAD
            self.save_plot_with_format(
                session_obj=st.session_state["panel_2_obj"],
                fig=fig,
                file_basename=file_name,
                file_name=f"Kinetic_{file_name}",
                button_key=f"panel2_{file_name}",
                selected_format=export_format
            )

=======
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
    def panel3(self):
        tmp_data_path = st.session_state.get("tmp_data_path")

        if "panel_3_obj" not in st.session_state or st.session_state["panel_3_obj"] is None:
            st.error("Panel 3 object not initialized. Please process data first.")
            return

        file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]

        with st.expander("Panel 3 - Contour Plot", expanded=True):
            st.markdown("# Contour Plot")

            with st.form(f"panel3_form_{file_name}"):
                zmin_zmax = st.slider(
                    "Select Zmin and Zmax",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get(f"panel3_zrange_{file_name}", (0.0, 1.0))
                )
                apply_panel3 = st.form_submit_button("Apply Panel 3 Settings")

            if apply_panel3 or f"panel3_zrange_{file_name}" not in st.session_state:
                st.session_state[f"panel3_zrange_{file_name}"] = zmin_zmax

            selected_zrange = st.session_state.get(f"panel3_zrange_{file_name}", (0.0, 1.0))

            contourplot = st.session_state["panel_3_obj"].plot(
                zmin=selected_zrange[0],
                zmax=selected_zrange[1]
            )

            st.pyplot(contourplot, clear_figure=False)

            self.save_matplotlib_formats(
                session_obj=st.session_state["panel_3_obj"],
                fig=contourplot,
                file_basename=file_name,
                file_name=f"Contour_{file_name}_{selected_zrange[0]}_{selected_zrange[1]}",
                button_key=f"panel3_{file_name}_{selected_zrange[0]}_{selected_zrange[1]}"
            )

    def panel4(self):
        tmp_ref_path = st.session_state.get("tmp_ref_path")

        if "panel_4_obj" not in st.session_state or st.session_state["panel_4_obj"] is None:
            st.warning("Panel 4: No reference data available.")
            return

        ref_file_name = os.path.splitext(os.path.basename(tmp_ref_path))[0]

        with st.expander("Panel 4 - Reference", expanded=True):
            st.markdown("# Reference")

            n_frames = st.session_state["panel_4_obj"].data.shape[1] - 1

            with st.form(f"panel4_form_{ref_file_name}"):
                i = self.safe_frame_slider("Select the frame for water reference", n_frames)
                apply_panel4 = st.form_submit_button("Apply Panel 4 Settings")

            if apply_panel4 or f"panel4_frame_{ref_file_name}" not in st.session_state:
                st.session_state[f"panel4_frame_{ref_file_name}"] = i

            selected_frame = st.session_state.get(f"panel4_frame_{ref_file_name}", 1)

            reference_plot = st.session_state["panel_4_obj"].plot(i=selected_frame)
            st.pyplot(reference_plot)

            self.save_matplotlib_formats(
                session_obj=st.session_state["panel_4_obj"],
                fig=reference_plot,
                file_basename=ref_file_name,
                file_name=f"Reference_{ref_file_name}_{selected_frame}",
                button_key=f"panel4_{ref_file_name}_{selected_frame}"
            )

    def panel6(self):
        data_path = st.session_state.get("tmp_data_path")

        if "panel_6_obj" not in st.session_state or st.session_state["panel_6_obj"] is None:
            st.error("Panel 6 object not initialized. Please process data first.")
            return

<<<<<<< HEAD
        with st.expander("Panel 6 - Stacked Spectra (Waterfall Plot)", expanded=True):
            st.markdown("# Stacked Spectra")

            col1, col2, col3 = st.columns(3)
=======
        file_name = os.path.splitext(os.path.basename(data_path))[0]
        panel6_obj = st.session_state["panel_6_obj"]
        max_timepoints = panel6_obj.n_timepoints

        with st.expander("Panel 6 - Waterfall Plot", expanded=True):
            st.markdown("# Waterfall Plot")

            with st.form(f"panel6_form_{file_name}"):
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)

                plot_mode = st.selectbox(
                    "Select waterfall display mode",
                    options=[
                        "Raw only",
                        "Fitted only",
                        "Raw + fitted",
                        "Diff only"
                    ],
                    index=[
                        "Raw only",
                        "Fitted only",
                        "Raw + fitted",
                        "Diff only"
                    ].index(st.session_state.get(f"panel6_mode_{file_name}", "Raw + fitted"))
                )

                time_range = st.slider(
                    "Select spectra/time-point range",
                    min_value=1,
                    max_value=max_timepoints,
                    value=st.session_state.get(
                        f"panel6_time_range_{file_name}",
                        (1, min(max_timepoints, 50))
                    )
                )

<<<<<<< HEAD
            with col3:
                spacing_factor = st.slider("Vertical spacing", 0.5, 3.0, 1.5, step=0.1)

            max_intensity = st.session_state["panel_6_obj"].data.iloc[:, 1:].max().max()
            spacing = max_intensity * spacing_factor

            fig = st.session_state["panel_6_obj"].plot_plotly(
                spacing=spacing,
                show_fit=show_fit,
                show_every_nth=show_every_nth
=======
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    show_every_nth = st.slider(
                        "Show every n-th timepoint",
                        1,
                        10,
                        st.session_state.get(f"panel6_every_{file_name}", 3)
                    )

                with col2:
                    azim = st.slider(
                        "Horizontal angle (°)",
                        min_value=0,
                        max_value=360,
                        value=st.session_state.get(f"panel6_azim_{file_name}", 45),
                        step=5
                    )

                with col3:
                    elev = st.slider(
                        "Vertical angle (°)",
                        min_value=5,
                        max_value=90,
                        value=st.session_state.get(f"panel6_elev_{file_name}", 25),
                        step=5
                    )

                with col4:
                    smooth = st.checkbox(
                        "Apply smoothing",
                        value=st.session_state.get(f"panel6_smooth_{file_name}", True)
                    )

                smooth_window = st.slider(
                    "Smoothing window",
                    3,
                    15,
                    st.session_state.get(f"panel6_smooth_window_{file_name}", 5),
                    step=2
                )

                apply_panel6 = st.form_submit_button("Apply Waterfall Settings")

            if apply_panel6 or f"panel6_ready_{file_name}" not in st.session_state:
                st.session_state[f"panel6_mode_{file_name}"] = plot_mode
                st.session_state[f"panel6_time_range_{file_name}"] = time_range
                st.session_state[f"panel6_every_{file_name}"] = show_every_nth
                st.session_state[f"panel6_azim_{file_name}"] = azim
                st.session_state[f"panel6_elev_{file_name}"] = elev
                st.session_state[f"panel6_smooth_{file_name}"] = smooth
                st.session_state[f"panel6_smooth_window_{file_name}"] = smooth_window
                st.session_state[f"panel6_ready_{file_name}"] = True

            selected_mode = st.session_state.get(f"panel6_mode_{file_name}", "Raw + fitted")
            selected_range = st.session_state.get(f"panel6_time_range_{file_name}", (1, min(max_timepoints, 50)))
            selected_every = st.session_state.get(f"panel6_every_{file_name}", 3)
            selected_azim = st.session_state.get(f"panel6_azim_{file_name}", 45)
            selected_elev = st.session_state.get(f"panel6_elev_{file_name}", 25)
            selected_smooth = st.session_state.get(f"panel6_smooth_{file_name}", True)
            selected_smooth_window = st.session_state.get(f"panel6_smooth_window_{file_name}", 5)

            fig_mpl = panel6_obj.plot_matplotlib_3d(
                azim=selected_azim,
                elev=selected_elev,
                plot_mode=selected_mode,
                time_start=selected_range[0],
                time_end=selected_range[1],
                show_every_nth=selected_every,
                smooth=selected_smooth,
                smooth_window=selected_smooth_window
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True}
            )

            file_name = os.path.splitext(os.path.basename(data_path))[0]

            fig_mpl = st.session_state["panel_6_obj"].plot_matplotlib(
                spacing=spacing,
                show_fit=show_fit,
                show_every_nth=show_every_nth
            )

<<<<<<< HEAD
            export_format = st.selectbox(
                "Choose export format",
                options=["PDF", "PNG", "JPG"],
                index=0,
                key=f"panel6_export_format_{file_name}_{show_fit}_{show_every_nth}_{spacing_factor}"
            )
=======
            approx_peaks = panel6_obj.validate_peak_positions()
            if approx_peaks:
                st.caption(
                    f"Approximate major peak positions across time (ppm): "
                    f"min={approx_peaks[0]:.2f}, median={approx_peaks[1]:.2f}, max={approx_peaks[2]:.2f}"
                )
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)

            self.save_matplotlib_formats(
                session_obj=panel6_obj,
                fig=fig_mpl,
                file_basename=file_name,
<<<<<<< HEAD
                file_name=f"Stacked_Spectra_{file_name}",
                button_key=f"panel6_{file_name}_{show_fit}_{show_every_nth}_{spacing_factor}",
                selected_format=export_format
=======
                file_name=(
                    f"Waterfall_{file_name}_"
                    f"{selected_mode.replace(' ', '_').replace('+', 'plus')}_"
                    f"{selected_range[0]}to{selected_range[1]}_"
                    f"azim{selected_azim}_elev{selected_elev}"
                ),
                button_key=(
                    f"panel6_{file_name}_{selected_mode}_"
                    f"{selected_range[0]}_{selected_range[1]}_"
                    f"{selected_every}_{selected_azim}_{selected_elev}_"
                    f"{selected_smooth}_{selected_smooth_window}"
                )
            )

    def about_page(self, about):
        with about:
            st.markdown(
                """
                ### Instructions

                1. Select model.
                2. Upload metadata, spectrum, and optional reference file.
                3. Press Start Processing.
                4. Enable only the panel you want to inspect.
                5. Change plot settings and click Apply.
                6. Select export format(s) and download results.

                ### Waterfall Plot Options

                - Raw only
                - Fitted only
                - Raw + fitted
                - Diff only
                - Select spectra/time-point range
                - Adjust viewing angles
                - Apply smoothing
                """
>>>>>>> 4da00a2 (Update waterfall plot features and UI improvements)
            )

    def run(self):
        self.header()


print("Layout.py loaded")
print("StreamlitApp:", "StreamlitApp" in dir())
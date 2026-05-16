import os
import tempfile
from pathlib import Path

import pandas as pd
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


class StreamlitApp:
    def __init__(self, fig1=None, fig2=None, fig3=None, fig4=None):
        self.fig1 = fig1
        self.fig2 = fig2
        self.fig3 = fig3
        self.fig4 = fig4

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
            else:
                st.session_state["Model 1"] = False
                st.session_state["Model 2"] = True

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

        if st.session_state["Model 1"] is True:
            from peak_fitting_v6 import PeakFitting
        else:
            from peak_fitting_v7 import PeakFitting

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
                self.process_batch(PeakFitting)
                if st.session_state.get("batch_results", {}).get("successful"):
                    self.process_plots()
            else:
                with st.spinner("Processing the data. Please wait..."):
                    self.process_data(PeakFitting)
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

                    st.info(
                        """
                        ### Batch Processing Complete

                        All results have been saved in the `output/` folder. Each file has its own subdirectory:
                        - `output/{filename}_output/`

                        To view individual results:
                        1. Switch off Batch Processing Mode
                        2. Upload a single file
                        3. View the analysis panels
                        """
                    )

                    if results["successful"]:
                        st.markdown("---")
                        st.markdown("### View Individual Results")
                        selected_file = st.selectbox(
                            "Select a processed file to view:",
                            options=results["successful"]
                        )

                        if st.button("Load Selected File"):
                            st.info(f"Loading {selected_file}... (This feature can be implemented)")
                else:
                    st.info("Click 'Start Processing' to process multiple files.")

            if st.session_state.get("processing_started", False):
                if (
                    st.session_state.get("file_name") is not None
                    or (st.session_state.get("batch_mode") and st.session_state.get("batch_results"))
                ):
                    self.panel1()
                    self.panel2()
                    self.panel3()
                    self.panel6()

                if st.session_state.get("panel_4_obj") is not None:
                    self.panel4()

                st.markdown("---")
                st.markdown("### Download Results")

                import shutil
                output_dir = os.path.abspath("output")

                if os.path.exists(output_dir):
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
                else:
                    st.warning("Output folder not found.")
            else:
                st.info("Click 'Start Processing' to see the analysis panels.")

    def process_data(self, PeakFitting):
        if not self.data_fp or not self.meta_fp:
            st.error("Please upload both the spectrum (.csv) and metadata (.xlsx) files before processing.")
            return

        original_data_name = self.data_fp.name
        original_meta_name = self.meta_fp.name

        tmp_data_dir = tempfile.gettempdir()
        tmp_data_path = os.path.join(tmp_data_dir, original_data_name)
        tmp_meta_path = os.path.join(tmp_data_dir, original_meta_name)

        with open(tmp_data_path, "wb") as f:
            f.write(self.data_fp.getbuffer())

        with open(tmp_meta_path, "wb") as f:
            f.write(self.meta_fp.getbuffer())

        if self.reference_fp:
            original_ref_name = self.reference_fp.name
            tmp_ref_path = os.path.join(tmp_data_dir, original_ref_name)
            with open(tmp_ref_path, "wb") as f:
                f.write(self.reference_fp.getbuffer())
        else:
            tmp_ref_path = None

        fitter = PeakFitting(tmp_data_path, tmp_meta_path)
        fitter.fit()

        fitting_params_path = os.path.join(fitter.output_direc, "fitting_params.csv")
        fitting_params_error_path = os.path.join(fitter.output_direc, "fitting_params_error.csv")

        fitter.fitting_params.to_csv(fitting_params_path)
        fitter.fitting_params_error.to_csv(fitting_params_error_path)

        processor = Process4Panels(tmp_data_path)
        processor.save_sum_spectra()
        processor.save_substrate_individual()
        processor.save_individual_peaks()
        processor.save_difference()
        processor.save_kinetics()

        st.session_state["tmp_data_path"] = tmp_data_path
        st.session_state["tmp_meta_path"] = tmp_meta_path
        st.session_state["tmp_ref_path"] = tmp_ref_path
        st.session_state["file_name"] = os.path.splitext(original_data_name)[0]

    def process_batch(self, PeakFitting):
        if not self.meta_fp or not self.data_files:
            st.error("Please upload metadata and at least one spectrum file.")
            return

        tmp_data_dir = tempfile.gettempdir()
        original_meta_name = self.meta_fp.name
        tmp_meta_path = os.path.join(tmp_data_dir, original_meta_name)

        with open(tmp_meta_path, "wb") as f:
            f.write(self.meta_fp.getbuffer())

        if self.reference_fp:
            original_ref_name = self.reference_fp.name
            tmp_ref_path = os.path.join(tmp_data_dir, original_ref_name)
            with open(tmp_ref_path, "wb") as f:
                f.write(self.reference_fp.getbuffer())
        else:
            tmp_ref_path = None

        total_files = len(self.data_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        successful_files = []
        failed_files = []

        for idx, data_file in enumerate(self.data_files):
            file_progress = (idx + 1) / total_files
            status_text.text(f"Processing file {idx + 1}/{total_files}: {data_file.name}")
            progress_bar.progress(file_progress)

            try:
                original_data_name = data_file.name
                tmp_data_path = os.path.join(tmp_data_dir, original_data_name)

                with open(tmp_data_path, "wb") as f:
                    f.write(data_file.getbuffer())

                with st.spinner(f"Fitting peaks for {data_file.name}..."):
                    fitter = PeakFitting(tmp_data_path, tmp_meta_path)
                    fitter.fit()

                    fitting_params_path = os.path.join(fitter.output_direc, "fitting_params.csv")
                    fitting_params_error_path = os.path.join(fitter.output_direc, "fitting_params_error.csv")
                    fitter.fitting_params.to_csv(fitting_params_path)
                    fitter.fitting_params_error.to_csv(fitting_params_error_path)

                with st.spinner(f"Processing data for {data_file.name}..."):
                    processor = Process4Panels(tmp_data_path)
                    processor.save_sum_spectra()
                    processor.save_substrate_individual()
                    processor.save_individual_peaks()
                    processor.save_difference()
                    processor.save_kinetics()

                if tmp_ref_path and st.session_state.get("use_reference", True):
                    try:
                        ref_processor = Reference(
                            fp_ref=tmp_ref_path,
                            fp_meta=tmp_meta_path,
                            fp_data=tmp_data_path
                        )
                        ref_processor.save_kinetics_mmol()
                    except Exception as e:
                        st.warning(f"Reference processing failed for {data_file.name}: {str(e)}")

                successful_files.append(data_file.name)

            except Exception as e:
                failed_files.append((data_file.name, str(e)))
                st.error(f"Failed to process {data_file.name}: {str(e)}")

        progress_bar.progress(1.0)
        status_text.empty()

        with results_container:
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

            if successful_files:
                summary_text = "Batch Processing Summary\n"
                summary_text += "=" * 50 + "\n\n"
                summary_text += f"Total files: {total_files}\n"
                summary_text += f"Successful: {len(successful_files)}\n"
                summary_text += f"Failed: {len(failed_files)}\n\n"

                summary_text += "Successful Files:\n"
                for file in successful_files:
                    summary_text += f"{file}\n"

                if failed_files:
                    summary_text += "\nFailed Files:\n"
                    for file, error in failed_files:
                        summary_text += f"Error: {file}: {error}\n"

                st.download_button(
                    label="Download Summary",
                    data=summary_text,
                    file_name="batch_processing_summary.txt",
                    mime="text/plain"
                )

        st.session_state["batch_results"] = {
            "successful": successful_files,
            "failed": failed_files,
            "total": total_files
        }

        if successful_files:
            last_file = successful_files[-1]
            st.session_state["tmp_data_path"] = os.path.join(tmp_data_dir, last_file)
            st.session_state["file_name"] = os.path.splitext(last_file)[0]
            st.session_state["tmp_meta_path"] = tmp_meta_path
            st.session_state["tmp_ref_path"] = tmp_ref_path

        st.session_state["processing_done"] = True

    def process_plots(self):
        tmp_data_path = st.session_state.get("tmp_data_path")
        tmp_meta_path = st.session_state.get("tmp_meta_path")
        tmp_ref_path = st.session_state.get("tmp_ref_path")
        use_reference = st.session_state.get("use_reference", True)

        if not tmp_data_path or not os.path.exists(tmp_data_path):
            st.error("Temporary data file not found. Please process data first.")
            return

        st.session_state["panel_1_obj"] = Panel1SpectrumPlot(file_path=tmp_data_path)
        st.session_state["panel_2_obj"] = KineticPlot(path=tmp_data_path)
        st.session_state["panel_3_obj"] = ContourPlot(file_path=tmp_data_path)

        if use_reference and tmp_ref_path and os.path.exists(tmp_ref_path):
            try:
                st.session_state["panel_4_obj"] = Reference(
                    fp_ref=tmp_ref_path,
                    fp_meta=tmp_meta_path,
                    fp_data=tmp_data_path
                )
                st.session_state["panel_4_obj"].save_kinetics_mmol()
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

    def panel1(self):
        try:
            tmp_data_path = st.session_state.get("tmp_data_path")
            if not tmp_data_path:
                st.error("No processed data found. Please process data first.")
                return

            file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]
            sum_fit_fp = Path("output", f"{file_name}_output", "sum_fit.csv")
            pd.read_csv(sum_fit_fp)

        except Exception as e:
            st.markdown(
                """
                <span style="color:red; font-size:72px;">Please Press 'Start Processing !'</span>
                """,
                unsafe_allow_html=True
            )
            st.error(f"Error loading data: {str(e)}")
            return

        if "panel_1_obj" not in st.session_state or st.session_state["panel_1_obj"] is None:
            st.error("Panel 1 object not initialized. Please process data first.")
            return

        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            panel1_obj = st.session_state["panel_1_obj"]
            data = panel1_obj.data
            diffs = panel1_obj.differences

            n_cols_data = data.shape[1]
            n_frames_data = max(n_cols_data - 1, 0)

            if n_frames_data < 1:
                st.info("Panel 1: No time frames found (only x-axis).")
                return

            time_frame = st.slider(
                "Select the frame",
                min_value=1,
                max_value=n_frames_data,
                value=1,
            )
            st.session_state["time_frame"] = time_frame

            st.markdown("# Substrate Plot")

            noise_std_text = "n/a"
            if isinstance(diffs, pd.DataFrame) and diffs.shape[1] > time_frame:
                noise_std = diffs.iloc[:, time_frame].std()
                noise_std_text = f"{noise_std:.3f}"

            st.write(f"Standard deviation of noise: {noise_std_text}")

            one_plot = panel1_obj.plot(time_frame)
            file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]

            st.plotly_chart(
                one_plot,
                use_container_width=True,
                config={"displayModeBar": True}
            )

            export_format = st.selectbox(
                "Choose export format",
                options=["JPG", "PNG", "PDF"],
                index=0,
                key=f"panel1_export_format_{file_name}_{time_frame}"
            )

            self.save_plot_with_format(
                session_obj=st.session_state["panel_1_obj"],
                fig=one_plot,
                file_basename=file_name,
                file_name=f"Substrate_{file_name}_{time_frame}",
                button_key=f"panel1_{file_name}_{time_frame}",
                selected_format=export_format
            )

    def panel2(self):
        if "panel_2_obj" not in st.session_state or st.session_state["panel_2_obj"] is None:
            st.error("Panel 2 object not initialized. Please process data first.")
            return

        tmp_data_path = st.session_state.get("tmp_data_path")
        if not tmp_data_path:
            st.error("No processed data found. Please process data first.")
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

            export_format = st.selectbox(
                "Choose export format",
                options=["JPG", "PNG", "PDF"],
                index=0,
                key=f"panel2_export_format_{file_name}"
            )

            self.save_plot_with_format(
                session_obj=st.session_state["panel_2_obj"],
                fig=fig,
                file_basename=file_name,
                file_name=f"Kinetic_{file_name}",
                button_key=f"panel2_{file_name}",
                selected_format=export_format
            )

    def panel3(self):
        if "panel_3_obj" not in st.session_state or st.session_state["panel_3_obj"] is None:
            st.error("Panel 3 object not initialized. Please process data first.")
            return

        tmp_data_path = st.session_state.get("tmp_data_path")
        if not tmp_data_path:
            st.error("No processed data found. Please process data first.")
            return

        with st.expander("Panel 3 - Contour Plot", expanded=True):
            st.markdown("# Contour Plot")
            zmin_zmax = st.slider("Select Zmin and Zmax", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
            contourplot = st.session_state["panel_3_obj"].plot(zmin=zmin_zmax[0], zmax=zmin_zmax[1])
            st.pyplot(contourplot, clear_figure=False)

            file_name = os.path.splitext(os.path.basename(tmp_data_path))[0]

            export_format = st.selectbox(
                "Choose export format",
                options=["PDF", "PNG", "JPG"],
                index=0,
                key=f"panel3_export_format_{file_name}_{zmin_zmax[0]}_{zmin_zmax[1]}"
            )

            self.save_plot_with_format(
                session_obj=st.session_state["panel_3_obj"],
                fig=contourplot,
                file_basename=file_name,
                file_name=f"Contour_{file_name}_{zmin_zmax[0]}_{zmin_zmax[1]}",
                button_key=f"panel3_{file_name}_{zmin_zmax[0]}_{zmin_zmax[1]}",
                selected_format=export_format
            )

    def panel4(self):
        if "panel_4_obj" not in st.session_state or st.session_state["panel_4_obj"] is None:
            st.warning("Panel 4: No reference data available. Please upload a reference file.")
            return

        tmp_ref_path = st.session_state.get("tmp_ref_path")
        if not tmp_ref_path:
            st.warning("No reference file uploaded.")
            return

        with st.expander("Panel 4 - Reference", expanded=True):
            st.markdown("# Reference")

            n_frames = st.session_state["panel_4_obj"].data.shape[1] - 1
            i = self.safe_frame_slider("Select the frame for water reference", n_frames)

            reference_plot = st.session_state["panel_4_obj"].plot(i=i)
            st.pyplot(reference_plot)

            ref_file_name = os.path.splitext(os.path.basename(tmp_ref_path))[0]

            export_format = st.selectbox(
                "Choose export format",
                options=["PDF", "PNG", "JPG"],
                index=0,
                key=f"panel4_export_format_{ref_file_name}_{i}"
            )

            self.save_plot_with_format(
                session_obj=st.session_state["panel_4_obj"],
                fig=reference_plot,
                file_basename=ref_file_name,
                file_name=f"Reference_{ref_file_name}_{i}",
                button_key=f"panel4_{ref_file_name}_{i}",
                selected_format=export_format
            )

    def panel6(self):
        data_path = st.session_state.get("tmp_data_path", None)
        panel6_obj = st.session_state.get("panel_6_obj", None)

        if panel6_obj is None:
            if data_path and os.path.exists(data_path):
                try:
                    st.session_state["panel_6_obj"] = StackedSpectraPlot(file_path=data_path)
                    panel6_obj = st.session_state["panel_6_obj"]
                except Exception as e:
                    st.error(f"Failed to initialize Panel 6 object: {e}")
                    return
            else:
                st.error("Panel 6 object not initialized. Please process data first.")
                return

        if not data_path:
            st.error("No processed data found. Please process data first.")
            return

        with st.expander("Panel 6 - Stacked Spectra (Waterfall Plot)", expanded=True):
            st.markdown("# Stacked Spectra")

            col1, col2, col3 = st.columns(3)

            with col1:
                show_fit = st.checkbox("Show fitted spectra", value=False)

            with col2:
                show_every_nth = st.slider("Show every n-th timepoint", 1, 10, 1)

            with col3:
                spacing_factor = st.slider("Vertical spacing", 0.5, 3.0, 1.5, step=0.1)

            max_intensity = st.session_state["panel_6_obj"].data.iloc[:, 1:].max().max()
            spacing = max_intensity * spacing_factor

            fig = st.session_state["panel_6_obj"].plot_plotly(
                spacing=spacing,
                show_fit=show_fit,
                show_every_nth=show_every_nth
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

            export_format = st.selectbox(
                "Choose export format",
                options=["PDF", "PNG", "JPG"],
                index=0,
                key=f"panel6_export_format_{file_name}_{show_fit}_{show_every_nth}_{spacing_factor}"
            )

            self.save_plot_with_format(
                session_obj=st.session_state["panel_6_obj"],
                fig=fig_mpl,
                file_basename=file_name,
                file_name=f"Stacked_Spectra_{file_name}",
                button_key=f"panel6_{file_name}_{show_fit}_{show_every_nth}_{spacing_factor}",
                selected_format=export_format
            )

    def run(self):
        self.header()


print("Layout.py wurde geladen")
print("StreamlitApp:", "StreamlitApp" in dir())
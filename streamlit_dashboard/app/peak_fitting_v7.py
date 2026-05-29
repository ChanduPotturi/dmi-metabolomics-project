import pandas as pd
import os
import re
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
from peakpos_df import SpectraAnalysis
import streamlit as st
import tempfile
import zipfile
import io

from peak_chunking import apply_metadata_chunking, label_from_metadata_column, log_chunking_stats


class PeakFitting:
    '''
    This class performs Lorentzian peak fitting on multi-frame NMR spectra using a 
    two-stage workflow:

    1. **Metadata-driven peak definition**  
       Peak positions and labels (substrate + metabolites + water) are extracted from 
       the metadata (.xlsx). These define the expected peaks for all frames.

    2. **Prefitting using actual spectral peak detection (peakpos_df.py)**  
       Before fitting the Lorentzian model, the class performs an automatic peak 
       detection step:
       
           SpectraAnalysis(self.df, self.positions)
       
       This adjusts the nominal metadata ppm values to actual observed peak locations 
       in each time step. As a result:
       
       • `self.positions` becomes a *list of lists*, containing detected peak positions  
         per time frame (not a single static list as in version 6).  
       • The first-stage “shared-shift model” from v6 is replaced by a direct fine-fit 
         using these pre-estimated peak centers.

    Fitting Model
    -------------
    For each time frame, the spectrum is modeled as a sum of Lorentzian curves:

        y(x) = Σ  A_k * gamma_k / ((x - x0_k)^2 + gamma_k^2)  +  y_shift

    where:
        x0_k   = peak centers (from prefitting)
        gamma_k = widths (shared per substance group)
        A_k     = amplitudes (shared per substance group)
        y_shift = vertical offset

    Constraints and bounds are defined in `make_bounds()`, providing:
        • frame-specific position bounds around detected peaks  
        • non-negative amplitudes  
        • controlled width range  
        • allowed shift offset  

    Output
    ------
    The class writes two CSV files into:

        <tmpdir>/<filename>_output/

    containing:
        • fitting_params.csv  
            └ y_shift, positions, widths, amplitudes per peak and time step  
        • fitting_params_error.csv  
            └ corresponding parameter uncertainties

    These files are consumed by downstream components such as:
        • Process4Panels (panel preprocessing)
        • KineticPlot (panel 2)
        • ContourPlot (panel 3)
        • StackedSpectraPlot (panel 6)
        • Reference (panel 4, mmol conversion)

    Differences Compared to Version 6
    ---------------------------------
    • Uses real peak detection (`SpectraAnalysis`) instead of metadata-only starting values  
    • No coarse global shift fit step — fitting starts directly from prefitted peak centers  
    • `self.positions` is now a 2D structure (one list per time frame)  
    • Improved NaN/inf cleaning before fitting  
    • Output directory moved to system temp folder  
    • Simplified bounds system for improved stability  
    • More robust handling of malformed CSV input  

    Parameters
    ----------
    fp_file : str
        Path to the multi-frame NMR spectrum (.csv)
    fp_meta : str
        Path to metadata (.xlsx)

    Notes
    -----
    • The first column of the CSV is expected to contain chemical shift values.  
    • Subsequent columns represent frames in a time series.  
    • All NaN and inf-containing rows are removed to avoid curve_fit failures.  
    • Prefitting significantly improves accuracy and convergence speed.  
    '''
    def __init__(
        self,
        fp_file,
        fp_meta,
        enable_chunking: bool = True,
        chunk_half_width: float = 0.5,
    ):
        # st.write('Using v7')
        # file paths
        self.fp_file = fp_file
        self.fp_meta = fp_meta

        # name of the data file and metadata file
        self.file_name = os.path.basename(fp_file)
        self.meta_name = os.path.basename(fp_meta)

        # define and create output directory
        self.output_direc = os.path.join(tempfile.gettempdir(), self.file_name + '_output')
        os.makedirs(self.output_direc, exist_ok=True)

        # load the spectral file
        self.df = pd.read_csv(fp_file, sep=None, engine="python", on_bad_lines="skip")

        # clean NaN and infinite values immediately after loading
        # first convert infinite values to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # remove all rows containing at least one NaN
        # (ensures self.df.iloc[:, i+1] never contains NaNs during fitting)
        before_shape = self.df.shape
        self.df.dropna(axis=0, how="any", inplace=True)
        after_shape = self.df.shape
        print(f"[PeakFitting v7] Cleaned data: {before_shape} -> {after_shape} (rows x cols)")

        # load metadata
        self.meta_df = pd.read_excel(fp_meta)

        # time point information
        self.number_time_points = self.df.shape[1] - 1
        self.time_points = np.arange(1, self.number_time_points + 1) 
        self.x = self.df.iloc[:,0]

        # positions and corresponding names of the peaks
        self.positions, self.names = self.extract_ppm_all()
        self.number_peaks = len(self.positions)
        self.number_substances = len(set(self.names))

        self.enable_chunking = enable_chunking
        self.chunk_half_width = chunk_half_width
        self._setup_chunking()

        # initialize output DataFrames
        column_names =  ['Time_Step'] + ['y_shift'] + [f'{name}_pos_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_width_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_amp_{pos}' for name, pos in zip(self.names, self.positions)]

        self.fitting_params = pd.DataFrame(columns=column_names)
        self.fitting_params_error = pd.DataFrame(columns=column_names)
        self.fitting_params['Time_Step'] = self.time_points
        self.fitting_params_error['Time_Step'] = self.time_points

        self.fitting_params = self.fitting_params.set_index('Time_Step')
        self.fitting_params_error = self.fitting_params_error.set_index('Time_Step')

        # internal mapping used for parameter grouping
        # (duplicate lists ensure grouping over substances)
        self.names_substances =  deepcopy(self.names) + list(dict.fromkeys(self.names)) + list(dict.fromkeys(self.names)) # Not a relevant instance attribute, so putting somewhere else?
        

        # prefitting step:
        # use spectral peak detection to refine initial peak positions
        fix_pos = SpectraAnalysis(self.df, self.positions)
        self.positions = fix_pos.peak_df()['Found Peaks']

    def _setup_chunking(self):
        """Metadata ppm → chunk windows → merge; restrict fitting to chunk mask."""
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
        """Return (x, y) arrays restricted to merged peak chunks when chunking is enabled."""
        if self.enable_chunking and hasattr(self, "fit_mask"):
            mask = self.fit_mask
            return self.x[mask].to_numpy(), y[mask].to_numpy()
        return self.x.to_numpy(), y.to_numpy()

    def extract_ppm_all(self):
        """
        Extract the ppm values from the metadata file for a specific file.

        Args:
            meta_df: metadata dataframe
            file_name: name of the file
        
        Returns:
            positions: list of all ppm values
            names: list of all names of the ppm values
        """

        def _normalize_file_name(value):
            text = str(value).strip().lower()
            # Treat "name _7.csv" and "name_7.csv" as the same file.
            text = re.sub(r"\s+_", "_", text)
            return text

        normalized_file_name = _normalize_file_name(self.file_name)
        self.meta_df = self.meta_df[
            self.meta_df['File'].astype(str).map(_normalize_file_name) == normalized_file_name
        ]

        if self.meta_df.empty:
            print(f"No metadata found for {self.file_name}")
            return [], []

        if self.meta_df.shape[0] == 0: #no metabolites listed --> only water present
            print(f'No metadata found for {self.file_name}')
            return [], []

        positions = []
        names = []

        substrate_name = label_from_metadata_column(
            self.meta_df, "Substrate", "Substrate"
        )

        react_substrat = str(self.meta_df['Substrate_ppm'].iloc[0]).split(',')
        if react_substrat and react_substrat != ['nan']:
            for val in react_substrat:
                names.append(substrate_name)
                positions.append(float(val))

        for i in range(1, 6):
            react_metabolite = str(self.meta_df[f'Metabolite_{i}_ppm'].iloc[0]).split(',')
            if react_metabolite == ['nan']:
                continue
            metabolite_name = label_from_metadata_column(
                self.meta_df, f"Metabolite_{i}", f"Metabolite {i}"
            )
            for val in react_metabolite:
                names.append(metabolite_name)
                positions.append(float(val))

        # water ppm
        positions.append(float(self.meta_df['Water_ppm'].iloc[0]))
        names.append(label_from_metadata_column(self.meta_df, "Water", "Water"))

        return positions, names

    def make_bounds(self, positions_fine = None,
                    y_shift = (0,np.inf),
                    shift_bounds_fine = (- 0.1, 0.1), width_bounds_fine = (0, 3e-1), amplitude_bounds_fine = (0, np.inf)):
        """
        Make the bounds for the fitting.

        Args:
            positions_fine: list of positions for fine tuning
            y_shift: tuple, lower and upper bound for the y_shift
            shift_bounds_fine: tuple, lower and upper bound for the shift in fine tuning
            width_bounds_fine: tuple, lower and upper bound for the width in fine tuning
            amplitude_bounds_fine: tuple, lower and upper bound for the amplitude in fine tuning

        Returns:
            numpy array: numpy array of the lower and upper bounds
        """
        y_shift_lower_bound = np.array([y_shift[0]])
        y_shift_upper_bound = np.array([y_shift[1]])
        shift_lower_bounds_fine = np.array(positions_fine) + shift_bounds_fine[0]
        # shift upper bounds fine-tun
        shift_upper_bounds_fine = np.array(positions_fine) + shift_bounds_fine[1]
        width_lower_bounds = np.full(self.number_substances, width_bounds_fine[0])
        width_upper_bounds = np.full(self.number_substances, width_bounds_fine[1])
        # amplitude lower bounds
        amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds_fine[0])
        # amplitude upper bounds
        amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds_fine[1])
        return (np.concatenate([y_shift_lower_bound,shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds]), np.concatenate([y_shift_upper_bound, shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds]))


    def unpack_params_errors(self, popt, pcov):
        """
        Unpack the parameters and errors from the fitting. This is because the fitting was done with shared parameters. To get the individual parameters, the shared parameters need to be unpacked.

        Args:
            n_unique_peaks: number of unique peaks
            number_peaks: number of peaks
            names: list of all names
            popt: fitted parameters
            pcov: covariance matrix
        
        Returns:
            tuple: tuple of the unpacked parameters and errors
        """
        error = np.sqrt(np.diag(pcov))
        # needs  n_unique_peaks, number_peaks, names, popt
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
            
        return np.concatenate([np.array([popt[0]]), popt[1:self.number_peaks+1], widths_final, amplitudes_final]), \
            np.concatenate([np.array([error[0]]), error[1:self.number_peaks+1], widths_final_error, amplitudes_final_error])

    def fit(self, save_csv = True):
        """
        Fit the data with the grey spectrum. The fitting parameters and errors are saved as csv files.

        Args:
            save_csv: bool, if True, the results are saved as csv files
        
        Returns:
            fitting_params: dataframe of the fitting parameters if save_csv is False
        """
        progress_bar = st.empty()
        load_bar = progress_bar.progress(0)
        first_fit = True
        # iterate over all time points
        for i in tqdm(range(self.number_time_points), desc= self.file_name):
            try: # try in case some data can not be fitted
                y = self.df.iloc[:,i+1]
                x_fit, y_fit = self._fitting_xy(y)
                # to increase fitting speed, increase tolerance 
                self.current_time_point = i
                # first fit kann weg
                if first_fit:
                    p0 = [0] + list(self.positions[i]) + [0.1]*self.number_substances + [1000]*self.number_substances
                    flattened_bounds_fine = self.make_bounds(positions_fine = list(self.positions[i]))
                    first_fit = False

                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: self.grey_spectrum_fine_tune(x, *params),
                                        x_fit, y_fit, p0 = p0, maxfev=20000, bounds = flattened_bounds_fine, ftol=1e-6, xtol=1e-6)
                
                y_shift = np.array([popt[0]])
                positions_fine = popt[1:self.number_peaks+1]
                widths = popt[self.number_peaks+1:self.number_peaks + self.number_substances+1]
                amplitudes = popt[self.number_peaks + self.number_substances+1:]
                
                p0 = np.concatenate([y_shift, np.array(self.positions[i]), widths, amplitudes])

                # unpack the parameters and errors
                self.fitting_params.loc[i+1], self.fitting_params_error.loc[i+1] = self.unpack_params_errors(popt, pcov)
                
            except RuntimeError:
                print(f'Could not fit time frame number {i}. Skipping...')
            load_bar.progress((i+1) / self.number_time_points)

        # remove loadbar
        progress_bar.empty()

        # set all NA values to 0, in case some time frames could not be fitted!
        self.fitting_params.fillna(0, inplace=True)
        self.fitting_params_error.fillna(0,inplace=True)

        # save results
        if save_csv == True:
            self.fitting_params.to_csv(self.output_direc + 'fitting_params.csv')
            self.fitting_params_error.to_csv(self.output_direc + 'fitting_params_error.csv')
        else:
            return self.fitting_params

    
    def lorentzian(self, x, shift, gamma, A):
        ''' 
        Lorentzian function

        Args:
            x: values to evaluate the function
            shift: shift parameter
            gamma: gamma parameter
            A: amplitude parameter

        Returns: 
            y:  calaculated values of the lorentzian function for x

        '''
        return A * gamma / ((x - shift)**2 + gamma**2)

    # this has high potential for being wrong 
    def grey_spectrum(self, x, *params):
        '''
        This method calculates the sum of lorentz function with shared widths and amplitudes.

        Args:
            x: values to evaluate the function
            params: list of parameters, first element is the y_shift, second element is the shift parameter, the next n elements are the gamma values, and the last n elements are the A values

        Returns:
            y: calculated values of the sum of the lorentzian functions
        '''
        shift = params[0]            # Single shift parameter
        gamma = params[1:self.number_substances+1]        # Extract n gamma values
        A = params[self.number_substances+1:]             # Extract n A values

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0] # not 0, it must be the first name from second ot third part of the mapping_names
        for i in range(self.number_peaks):
            # retrieve gamma and A values
            # Peak position is shared between all peaks
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]
            if k < self.number_peaks:
                y += self.lorentzian(x, shift + self.positions[self.current_time_point][i], gamma[k], A[k])
        # stop code from execution
        return y
    
    def write_results(self):
        '''
        Save the fitting parameters and errors as csv files.
        '''
        self.fitting_params.to_csv(self.output_direc + 'fitting_params.csv')
        self.fitting_params_error.to_csv(self.output_direc + 'fitting_params_error.csv')

    # this has high potential for being wrong
    def grey_spectrum_fine_tune(self, x, *params):
        '''
        This method calculates the sum of lorentz function with shared widths and amplitudes.

        Args:
            x: values to evaluate the function
            params: list of parameters, first element is the y_shift, second element is the shift parameter, the next n elements are the gamma values, and the last n elements are the A values

        Returns:
            y: calculated values of the sum of the lorentzian functions
        '''
        y_shift = params[0]
        x0 = params[1:self.number_peaks+1]
        gamma = params[1+self.number_peaks:self.number_peaks + self.number_substances+1]
        A = params[self.number_peaks+self.number_substances + 1:]

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

    def zip_output_directory(output_dir_path):
        """
        Zips all files in the given output directory and returns the in-memory zip file as BytesIO.
        """
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(output_dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir_path)  # relative name inside the zip
                    zip_file.write(file_path, arcname)

        zip_buffer.seek(0)
        return zip_buffer
import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
import streamlit as st
import tempfile


class PeakFitting:
    """
    Fits NMR spectra with Lorentzian peak models based on metadata-defined peak positions.

    This class:
    - loads a time-resolved spectrum from CSV,
    - finds the matching metadata row for the file,
    - extracts peak positions for substrate, metabolites and water,
    - performs a two-stage Lorentzian peak fit (coarse + fine),
    - stores fit parameters (position, width, amplitude, y-shift) and their errors
      for each time step.

    Attributes:
        fp_file (str): Path to the spectrum CSV file.
        fp_meta (str): Path to the metadata Excel file.
        file_name (str): Base name of the spectrum file (without extension).
        meta_name (str): Base name of the metadata file.
        output_direc (str): Output directory for fit results.
        df (pandas.DataFrame): Cleaned spectrum data (first column = x, others = timepoints).
        meta_df (pandas.DataFrame): Filtered metadata row for the current file.
        number_time_points (int): Number of time frames in the spectrum.
        time_points (numpy.ndarray): Array of time step indices (1..N).
        x (pandas.Series): x-axis values (chemical shifts).
        positions (list[float]): Peak positions (ppm) from metadata.
        names (list[str]): Peak labels (e.g. 'ReacSubs', 'Metab1', 'Water') in same order as positions.
        number_peaks (int): Total number of peaks (len(positions)).
        number_substances (int): Number of distinct peak groups (unique names).
        fitting_params (pandas.DataFrame): Fitted parameters per time step.
        fitting_params_error (pandas.DataFrame): Estimated parameter errors per time step.
        names_substances (list[str]): Helper list for grouping shared parameters across peaks.

    Methods:
        extract_ppm_all():
            Reads ppm positions for substrate, metabolites and water from metadata.

        make_bounds(...):
            Constructs lower and upper bounds for curve fitting for either coarse
            ('first') or fine-tuning ('fine') modes.

        unpack_params_errors(popt, pcov):
            Unpacks shared fit parameters and reconstructs per-peak widths and amplitudes,
            including propagated errors.

        fit(save_csv=True):
            Performs the full two-stage fit for all time steps and optionally saves
            `fitting_params` and `fitting_params_error` as CSV files.

        lorentzian(x, shift, gamma, A):
            Evaluates a single Lorentzian peak.

        grey_spectrum(x, *params):
            Computes the sum of Lorentzian peaks in the coarse (shared) parameterization.

        grey_spectrum_fine_tune(x, *params):
            Computes the sum of Lorentzian peaks in the fine-tuned parameterization
            with individual positions.

        write_results():
            Convenience method to save `fitting_params` and `fitting_params_error`
            as CSV files in the output directory.
    """

    def __init__(self, fp_file, fp_meta):
        # file paths
        self.fp_file = fp_file
        self.fp_meta = fp_meta
        # name of the data file and metadata file
        self.file_name = os.path.splitext(os.path.basename(fp_file))[0]
        self.meta_name = os.path.basename(fp_meta)

        # define and create output directory
        self.output_direc = os.path.join("output", self.file_name + "_output")
        os.makedirs(self.output_direc, exist_ok=True)

        # read in the files
        # use delimiter sniffing to support both comma/semicolon CSVs and skip malformed rows from instrument exports
        self.df = pd.read_csv(fp_file, sep=None, engine="python", on_bad_lines="skip")

        # normalize European decimal commas to dots and coerce to numeric
        for col in self.df.columns:
            # string replace only on object columns to avoid touching already-numeric cols
            if self.df[col].dtype == object:
                self.df[col] = self.df[col].astype(str).str.replace(",", ".", regex=False)
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # drop rows with NaNs or infs that would break curve_fit
        before_shape = self.df.shape
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=0, how="any", inplace=True)
        after_shape = self.df.shape
        if before_shape != after_shape:
            print(f"[PeakFitting] Cleaned data: {before_shape} -> {after_shape} (rows x cols)")
        self.meta_df = pd.read_excel(fp_meta)

        print("Loaded metadata columns:", self.meta_df.columns.tolist())
        print("Loaded metadata entries:", self.meta_df["File"].astype(str).tolist())

        self.number_time_points = self.df.shape[1] - 1
        self.time_points = np.arange(1, self.number_time_points + 1) 
        self.x = self.df.iloc[:,0]

        # positions and corresponding names of the peaks
        self.positions, self.names = self.extract_ppm_all()
        self.number_peaks = len(self.positions)
        self.number_substances = len(set(self.names))

         # initialize outputs
        column_names =  ['Time_Step'] + ['y_shift'] + [f'{name}_pos_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_width_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_amp_{pos}' for name, pos in zip(self.names, self.positions)]

        self.fitting_params = pd.DataFrame(columns=column_names)
        self.fitting_params_error = pd.DataFrame(columns=column_names)

        # set times
        self.fitting_params['Time_Step'] = self.time_points
        self.fitting_params_error['Time_Step'] = self.time_points
        
        self.fitting_params = self.fitting_params.set_index('Time_Step')
        self.fitting_params_error = self.fitting_params_error.set_index('Time_Step')

        self.names_substances =  deepcopy(self.names) + list(dict.fromkeys(self.names)) + list(dict.fromkeys(self.names)) # Not a relevant instance attribute, so putting somewhere else?
        
    def extract_ppm_all(self):
        """
        Extracts peak positions (ppm) and corresponding names from the metadata
        for the current file.

        The method:
        - normalizes the 'File' column in the metadata (lowercase, stripped),
        - finds the row whose 'File' entry contains the current `file_name`,
        - parses substrate and metabolite ppm columns,
        - adds the water peak position (from metadata or fallback to 4.7 ppm).

        If no matching metadata row is found, only a single water peak at 4.7 ppm
        is returned.

        Returns:
            tuple[list[float], list[str]]:
                positions: list of all ppm values
                names: list of corresponding peak labels
        """

        # Standardization of filter names
        filename_base = os.path.splitext(self.file_name)[0].strip().lower()

        # Standardize capitalization and spacing
        self.meta_df["File"] = self.meta_df["File"].astype(str).str.strip().str.lower()

        # Filter by file name
        filtered = self.meta_df[self.meta_df["File"].str.contains(filename_base, na=False)]

        # If no line was found:
        if filtered.empty:
            st.warning(f"No metadata found for file '{self.file_name}'. Only water peak will be fitted.")
            return [4.7], ["Water"]  # Standard-Wasserpeak in ppm

        # Apply metadata row
        self.meta_df = filtered.reset_index(drop=True)

        positions = []
        names = []

        # Substrat-PPM
        try:
            react_substrat = str(self.meta_df['Substrate_ppm'].iloc[0]).split(',')
            if react_substrat != ['nan']:
                for val in react_substrat:
                    names.append('ReacSubs')
                    positions.append(float(val))
        except Exception as e:
            print("No substrate found:", e)

        # Metabolites
        for i in range(1, 6):
            try:
                react_metabolite = str(self.meta_df[f'Metabolite_{i}_ppm'].iloc[0]).split(',')
                if react_metabolite != ['nan']:
                    for val in react_metabolite:
                        names.append(f'Metab{i}')
                        positions.append(float(val))
            except Exception:
                continue

        # Add water (always available)
        try:
            positions.append(float(self.meta_df['Water_ppm'].iloc[0]))
            names.append('Water')
        except Exception:
            positions.append(4.7)
            names.append('Water')

        return positions, names 


    def make_bounds(self, mode, positions_fine = None,
                    y_shift = (0, np.inf),
                    shift_bounds = (-np.inf, np.inf),width_bounds = (0, 3e-1), amplitude_bounds = (0, np.inf),
                    shift_bounds_fine = (- 0.1, 0.1), width_bounds_fine = (0, 3e-1), amplitude_bounds_fine = (0, np.inf)):
        """
        Make the bounds for the fitting. The bounds are different for the first fitting and the fine tuning.

        Args:

            mode: str, 'first' or 'fine'
            positions_fine: list of positions for fine tuning
            y_shift: tuple, lower and upper bound for the y_shift
            shift_bounds: tuple, lower and upper bound for the shift
            width_bounds: tuple, lower and upper bound for the width
            amplitude_bounds: tuple, lower and upper bound for the amplitude
            shift_bounds_fine: tuple, lower and upper bound for the shift in fine tuning
            width_bounds_fine: tuple, lower and upper bound for the width in fine tuning
            amplitude_bounds_fine: tuple, lower and upper bound for the amplitude in fine tuning

        Returns:
            numpy array: numpy array of the lower and upper bounds
        """
        if mode == 'first':
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds = np.full(1, shift_bounds[0])  # shifting the whole spectrum
            shift_upper_bounds = np.full(1, shift_bounds[1])
            width_lower_bounds = np.full(self.number_substances, width_bounds[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds[1])
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds[0])
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds[1])
            return (np.concatenate([y_shift_lower_bounds,shift_lower_bounds, width_lower_bounds, amplitude_lower_bounds]), np.concatenate([y_shift_upper_bounds,shift_upper_bounds, width_upper_bounds, amplitude_upper_bounds]))
        
        elif mode == 'fine':
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds_fine = positions_fine + shift_bounds_fine[0]
            # shift upper bounds fine-tun
            shift_upper_bounds_fine = positions_fine + shift_bounds_fine[1]
            width_lower_bounds = np.full(self.number_substances, width_bounds_fine[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds_fine[1])
            # amplitude lower bounds
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds_fine[0])
            # amplitude upper bounds
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds_fine[1])

            return (np.concatenate([y_shift_lower_bounds, shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds]), np.concatenate([y_shift_upper_bounds, shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds]))


    def unpack_params_errors(self, popt, pcov):
        """
        Unpacks shared fit parameters and their errors into per-peak values.

        The fitting uses shared widths and amplitudes per substance group.
        This method reconstructs individual widths and amplitudes for all peaks
        based on `self.names`, `self.number_peaks` and `self.number_substances`.

        Args:
            popt (numpy.ndarray): Optimized fit parameters from `curve_fit`.
            pcov (numpy.ndarray): Covariance matrix from `curve_fit`.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                - flattened parameter vector [y_shift, positions..., widths..., amplitudes...]
                - corresponding 1D error vector in the same order.
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
        Fit the data with the grey spectrum. The fitting is done in two steps. First, the whole spectrum is fitted with shared parameters. Second, the parameters are fine tuned.
        The fitting parameters and errors are saved as csv files.

        Args:
            save_csv: bool, if True, the results are saved as csv files
        
        Returns:
            fitting_params: dataframe of the fitting parameters if save_csv is False
        """

        # bounds for the first fitting, which corresponds to the first frame
        flattened_bounds = self.make_bounds(mode='first')
        
        progress_bar = st.empty()
        load_bar = progress_bar.progress(0)
        first_fit = True
        # iterate over all time points
        for i in tqdm(range(self.number_time_points), desc= self.file_name):
            try: # try in case some data can not be fitted
                y = self.df.iloc[:,i+1]
                # to increase fitting speed, increase tolerance 
                if first_fit:
                    popt, pcov = curve_fit(lambda x, *params: self.grey_spectrum(x,*params),
                                        self.x, y, p0=[0] + [0] + [0.1]*self.number_substances + [1000]*self.number_substances,
                                        maxfev=3000, ftol=1e-1, xtol=1e-1, bounds=flattened_bounds)
                    
                    y_shift = np.array([popt[0]])
                    # init parameters for fine tuning
                    positions_fine = popt[1] + self.positions
                    widths = popt[2:self.number_substances+2] 
                    amplitudes = popt[self.number_substances+2:]

                    # starting parameters for fine tuning
                    p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])
                    # bounds for fine tuning
                    flattened_bounds_fine = self.make_bounds(mode = 'fine', positions_fine = positions_fine)
                    first_fit = False

                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: self.grey_spectrum_fine_tune(x, *params),
                                        self.x, y, p0 = p0, maxfev=20000, bounds = flattened_bounds_fine, ftol=1e-6, xtol=1e-6)

                y_shift = np.array([popt[0]])
                positions_fine = popt[1:self.number_peaks+1]
                widths = popt[1+self.number_peaks:self.number_peaks + self.number_substances+1]
                amplitudes = popt[self.number_peaks + self.number_substances+1:]

                p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])
                
                # unpack the parameters and errors
                self.fitting_params.loc[i + 1], self.fitting_params_error.loc[i + 1] = self.unpack_params_errors(popt, pcov)
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

    def grey_spectrum(self, x, *params):
        '''
        This method calculates the sum of lorentz function with shared widths and amplitudes.

        Args:
            x: values to evaluate the function
            params: list of parameters, first element is the y_shift, second element is the shift parameter, the next n elements are the gamma values, and the last n elements are the A values

        Returns:
            y: calculated values of the sum of the lorentzian functions
        '''
        y_shift = params[0]           
        shift = params[1]            # Single shift parameter
        gamma = params[2:self.number_substances+2]        # Extract n gamma values
        A = params[self.number_substances+2:]             # Extract n A values

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0]
        for i in range(self.number_peaks):
            # retrieve gamma and A values
            # Peak position is shared between all peaks
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]
            if k < self.number_peaks:
                y += self.lorentzian(x, shift + self.positions[i], gamma[k], A[k]) + y_shift
        # stop code from execution
        return y
    
    def write_results(self):
        '''
        Save the fitting parameters and errors as csv files.
        '''
        self.fitting_params.to_csv(os.path.join(self.output_direc, 'fitting_params.csv'))
        self.fitting_params_error.to_csv(os.path.join(self.output_direc, 'fitting_params_error.csv'))

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
        gamma = params[1+ self.number_peaks:self.number_peaks + self.number_substances+1]
        A = params[self.number_peaks+self.number_substances+1:]

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

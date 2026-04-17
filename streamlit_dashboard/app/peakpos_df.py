import pandas as pd
import numpy as np


class SpectraAnalysis:
    """
    Helper class to detect peaks in NMR spectra using derivatives, prior to
    running the Lorentzian fit in Model 2.

    Workflow:
        1. Optionally normalize spectra so that the water peak is aligned to 4.7 ppm.
        2. Sum spectra over time and use derivative-based criteria to detect peaks.
        3. Match detected peaks to expected peak positions from metadata.
        4. Track peak positions over sections of the time series.
    """
    def __init__(self, data, expected_peaks):
        self.data = data
        self.expected_peaks = expected_peaks
        self.spectra_data = data.iloc[:, 1:]
        self.chem_shifts = data.iloc[:, 0]

    def peakfit_sum(self, threshold):
        """
        Sums all spectra and returns a list of found peaks above a given percentile threshold.

        The method:
        - sums all spectra over time,
        - computes first to fourth derivatives,
        - identifies peak candidates based on:
            * intensity above a percentile threshold,
            * negative second derivative (local maxima),
            * positive fourth derivative,
            * sign change in the third derivative.

        Args:
            threshold (float): Percentile of the summed intensities used as a
                               minimum intensity threshold (e.g. 85 for 85th percentile).

        Returns:
            list[float]: List of detected peak positions (in ppm).
        """

        sum_of_spectra = np.sum(self.spectra_data, axis=1)
        threshold = np.percentile(sum_of_spectra, threshold)

        first_derivative = np.gradient(sum_of_spectra, self.chem_shifts)
        second_derivative = np.gradient(first_derivative, self.chem_shifts)
        third_derivative = np.gradient(second_derivative, self.chem_shifts)
        fourth_derivative = np.gradient(third_derivative, self.chem_shifts)
        
        #calculate minima below zero in 2nd derivative
        sign_change = np.diff(np.sign(third_derivative)) != 0
        peak_mask = (sum_of_spectra > threshold) & (second_derivative < 0) & (fourth_derivative > 0)
        peak_mask[1:] &= sign_change

        #return list
        peak_pos = self.chem_shifts[peak_mask].tolist()
        return peak_pos

    def normalize_water(self):
        """
        Shifts the chemical shift axis so that the water peak is aligned to 4.7 ppm.

        The method:
        - detects prominent peaks in the summed spectrum,
        - finds the peak closest to 4.7 ppm,
        - computes the required shift,
        - returns a copy of the data with the chemical shift column adjusted.

        Returns:
            tuple:
                data_normalized (pandas.DataFrame): Copy of the original data with
                    the chemical shift column shifted so that water ≈ 4.7 ppm.
                norm_shift (float): Applied shift (4.7 - detected_water_peak).
        """

        #85th percentile as threshold
        peak_pos = self.peakfit_sum(threshold = 85)
        water = 4.7

        #select peal closest to 4.7 and calculate shift
        closest_peak = min(peak_pos, key=lambda x: abs(x - water))
        norm_shift = (4.7 - closest_peak)

        #shift spectra
        data_normalized = pd.DataFrame(self.chem_shifts.copy() + norm_shift)
        data_normalized = pd.concat([data_normalized, self.data.iloc[:, 1:]], axis=1)
        data_normalized.columns = self.data.columns
        return data_normalized, norm_shift

    def peak_identify(self, data_normalized, reference_peaks, initial_threshold=85, max_shift=0.1):
        """
        Detects peaks and assigns them to expected reference positions.

        The algorithm:
        - repeatedly detects peaks in the normalized data,
        - starts with a relatively high percentile threshold and lowers it stepwise,
        - matches detected peaks to the closest expected peak positions (reference_peaks),
        - accepts matches if the distance is below `max_shift`,
        - collects unmatched peaks as "other",
        - stops when all peaks are found, the threshold becomes too low, or
          too many unknown peaks are detected.

        Args:
            data_normalized (pandas.DataFrame): Normalized spectrum data
                (output of `normalize_water`).
            reference_peaks (list[float]): Expected peak positions in ppm, typically
                previous section's result or the initial `self.expected_peaks`.
            initial_threshold (int, optional): Starting percentile threshold for
                peak detection. Defaults to 85.
            max_shift (float, optional): Maximum allowed chemical shift deviation
                from the expected peak position for assignment. Defaults to 0.1.

        Returns:
            tuple:
                found (list[float]): List of actual peak positions in the same order
                    as `reference_peaks`.
                other (list[float]): List of detected peaks that could not be assigned
                    to any expected peak within `max_shift`.
        """
        spectra_data = data_normalized.iloc[:, 1:]
        chem_shifts = data_normalized.iloc[:, 0]

        found = [None] * len(reference_peaks)
        other = []
        threshold = initial_threshold

        while None in found or len(other) <= len(found):
            #search for peaks
            detected_peaks = self.peakfit_sum(threshold)

            #match found peaks to expected peaks
            for peak in detected_peaks:
                distances = [abs(peak - expected_peak) for expected_peak in reference_peaks]
                min_distance = min(distances)
                index = distances.index(min_distance)

                if min_distance <= max_shift:
                    if found[index] is None:
                        found[index] = peak
                    elif found[index] != peak:
                        other.append(peak)
                else:
                    other.append(peak)

            #lower threshold 
            threshold -= 2
            if threshold < 0 or None not in found or len(other) > len(found):
                break
        
        # replace remaining unfound peak positions with expected chem_shift value 
        for i, peak in enumerate(found):
            if peak is None:
                found[i] = reference_peaks[i]  


        return found, other

    def peak_df(self, min_cols_per_section=5, max_shift=None):
        """
        Normalizes spectra to water = 4.7 ppm, splits them into time sections and
        identifies peak positions per section.

        Steps:
            1. Normalize data using `normalize_water()`.
            2. Split the time series into sections with at least `min_cols_per_section` columns.
            3. For each section:
                - Detect peaks using `peak_identify`.
                - Match peaks to the expected positions from the previous section.
            4. Optionally compute a maximum shift between sections if `max_shift` is not provided.
            5. Undo the water normalization shift for the final peak positions.

        Args:
            min_cols_per_section (int, optional): Minimum number of spectra per section.
                Defaults to 5.
            max_shift (float, optional): Maximum allowed shift in ppm for peak matching
                between sections. If None, it is derived from an assumed total maximum
                shift of 0.15 ppm over the entire experiment (scaled by number of sections).

        Returns:
            pandas.DataFrame:
                final_df with columns:
                    - 'Column': original spectrum column names (excluding chemical shift).
                    - 'section': section index for each column.
                    - 'Found Peaks': list of detected peak positions for that column
                                     (aligned with expected peaks).
                    - 'Other Peaks': list of additional, unidentified peaks.
        """
        #normalize data
        df, norm_shift = self.normalize_water()
    
        # Split into section with at least 5 columns each
        cols_to_divide = len(df.columns)-1 #-1 because first column is chemical shift
        num_sections = max(cols_to_divide // min_cols_per_section, 1) if cols_to_divide > min_cols_per_section else 1

        cols_per_section = cols_to_divide // num_sections
        extra_cols = cols_to_divide % num_sections  # Extra columns to distribute

        sections = []
        start_col = 1 #ignore first column
        col_sect = []

        for section in range(num_sections):
            extra = 1 if section < extra_cols else 0
            end_col = start_col + cols_per_section + extra
            sections.append(df.iloc[:, [0] + list(range(start_col, end_col))]) #add column with chem_shift to each section
            start_col = end_col
            col_sect.extend([f"{section + 1}"] * (cols_per_section + extra)) #column entries for final dataframe

        reference_peaks = self.expected_peaks
        found_lists = []
        other_lists = []

        #create maximum shift based on assumption, that peaks may shift up to 0.1 during whole experiment. 
        # (1.5 to account for uneven distribution of pH-shifts)
        if max_shift == None:
            max_shift = 0.15 / num_sections 
        
        # Find peaks for each section,max_shift concerning previously found position if exist
        for df_section in sections:
            found, other = self.peak_identify(data_normalized = df_section, reference_peaks = reference_peaks, max_shift= max_shift)
            found_lists.append(found)
            other_lists.append(other)
            reference_peaks = found

        #'unnormalize' the values
        for i in range(len(found_lists)):
            found_lists[i] = [x - norm_shift for x in found_lists[i]]
            other_lists[i] = [x - norm_shift for x in other_lists[i]]

        #create final dataframe
        final_df = pd.DataFrame({
            'Column': df.columns[1:],  # Exclude the chemical shift from the final DataFrame columns
            'section': col_sect
        })

        peaks_data = {'Found Peaks': [], 'Other Peaks': []}

        for i in range(num_sections):
            section_length = len(sections[i].columns) - 1  # -1 to ignore the chemical shift column
            peaks_data['Found Peaks'].extend([found_lists[i]] * section_length)
            peaks_data['Other Peaks'].extend([other_lists[i]] * section_length)


        final_df['Found Peaks'] = peaks_data['Found Peaks']
        final_df['Other Peaks'] = peaks_data['Other Peaks']

        return final_df
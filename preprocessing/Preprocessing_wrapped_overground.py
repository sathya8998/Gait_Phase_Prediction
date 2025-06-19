import csv
from fractions import Fraction
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import sosfilt, butter
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import scipy.io as sio
from Gait_Phase.gpu_utils import configure_gpu

configure_gpu()

def align_emg_data_pipeline():
    #============================================================================
    # 1. USER PARAMETERS & FILE PATHS
    #============================================================================
    file_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Take+2024-10-17+07.08.33+PM_XYZ.csv'
    trigger_file_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\trigger_signal.mat"
    emg_files = [
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi1_EMG.mat",
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi2_EMG.mat",
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi3_EMG.mat",
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi+2_EMG.mat",
    ]
    
    marker_headers_row = 3
    data_start_row = 7
    AXIS_ORDER = ["X", "Y", "Z"]
    INTERPOLATION_METHOD = "linear"
    SKIP_UNLABELED = True
    ZERO_MARKER = "RCA"  # Marker to recenter coordinates on
    
    #============================================================================
    # 2. READ THE RAW CSV LINES
    #============================================================================
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        lines = list(reader)
    
    if marker_headers_row >= len(lines) or data_start_row >= len(lines):
        raise ValueError("Header or data start rows exceed total lines in file.")
    
    #============================================================================
    # 3. DYNAMICALLY FIND STARTING COLUMN FOR MARKER DATA
    #============================================================================
    raw_header = lines[marker_headers_row]
    STARTING_COLUMN = None
    for idx, col in enumerate(raw_header):
        if "Sathy:LASIS" in col:
            STARTING_COLUMN = idx
            break
    
    if STARTING_COLUMN is None:
        raise ValueError("Could not find starting marker column (e.g., 'Sathy:LASIS').")
    
    print(f"'Sathy:LASIS' found at column index {STARTING_COLUMN}")
    raw_marker_labels = raw_header[STARTING_COLUMN:]
    num_raw_cols = len(raw_marker_labels)
    print(f"Detected {num_raw_cols} marker columns starting from index {STARTING_COLUMN}.")
    
    num_groups = num_raw_cols // 3
    if num_raw_cols % 3 != 0:
        print("WARNING: Number of marker columns is not a multiple of 3. Some data may be ignored.")
    
    #============================================================================
    # 4. PARSE MARKER LABELS
    #============================================================================
    marker_names = []
    group_indices = []
    for g in range(num_groups):
        cX, cY, cZ = 3 * g, 3 * g + 1, 3 * g + 2
        label_x = raw_marker_labels[cX].strip()
        label_y = raw_marker_labels[cY].strip()
        label_z = raw_marker_labels[cZ].strip()
    
        def strip_final_axis(lab):
            for ax in [":X", ":Y", ":Z"]:
                if lab.endswith(ax):
                    return lab[:-2].strip()
            return lab
    
        name_x = strip_final_axis(label_x)
        name_y = strip_final_axis(label_y)
        name_z = strip_final_axis(label_z)
    
        if name_x == name_y == name_z:
            marker_name = name_x
        else:
            raise ValueError(f"Inconsistent marker naming: {label_x}, {label_y}, {label_z}")
    
        if SKIP_UNLABELED and "Unlabeled" in marker_name:
            continue
    
        marker_names.append(marker_name)
        group_indices.append((cX, cY, cZ))
    
    print(f"Kept {len(marker_names)} markers.")
    
    final_headers = []
    for mname in marker_names:
        safe_name = mname.replace(":", "_")
        for ax in AXIS_ORDER:
            final_headers.append(f"{safe_name}_{ax}")
    
    #============================================================================
    # 5. EXTRACT NUMERIC DATA AND FRAME/TIME
    #============================================================================
    data_rows = lines[data_start_row:]
    frame_col = [int(row[0].strip()) for row in data_rows]
    time_col = [float(row[1].strip()) for row in data_rows]
    
    all_numeric_rows = []
    for row in data_rows:
        slice_ = row[STARTING_COLUMN:]
        float_vals = []
        for (ix, iy, iz) in group_indices:
            valx = slice_[ix] if ix < len(slice_) else ""
            valy = slice_[iy] if iy < len(slice_) else ""
            valz = slice_[iz] if iz < len(slice_) else ""
    
            def to_float_or_nan(v):
                try:
                    return float(v.strip()) if v.strip() != "" else np.nan
                except:
                    return np.nan
    
            float_vals.extend([to_float_or_nan(valx), to_float_or_nan(valy), to_float_or_nan(valz)])
        all_numeric_rows.append(float_vals)
    
    df = pd.DataFrame(all_numeric_rows, columns=final_headers)
    df.insert(0, "Frame", frame_col)
    df.insert(1, "Time (Seconds)", time_col)
    
    #============================================================================
    # 6. INTERPOLATE MISSING VALUES
    #============================================================================
    df = df.interpolate(method=INTERPOLATION_METHOD, limit_direction="both", axis=0)
    df = df.round(6)
    print(f"DataFrame shape after interpolation: {df.shape}")
    
    #============================================================================
    # 7. LOAD TRIGGER SIGNAL AND EXTRACT SIMPLE START/END TIMES
    #============================================================================
    trigger_data = sio.loadmat(trigger_file_path)
    trigger_signal = trigger_data['Data'][0, 0].squeeze()
    trigger_time = trigger_data['Time'][0, 0].squeeze()
    
    trigger_start = trigger_time[0]
    trigger_end = trigger_time[-1]
    print(f"Trigger start time: {trigger_start}, end time: {trigger_end}")
    
    #============================================================================
    # 8. ALIGN AND TRIM MARKER DATA BASED ON TRIGGER SIGNAL
    #============================================================================
    aligned_markers_data = df[
        (df["Time (Seconds)"] >= trigger_start) &
        (df["Time (Seconds)"] <= trigger_end)
    ].copy()
    
    # Remove duplicate columns to avoid reindexing issues
    aligned_markers_data = aligned_markers_data.loc[:, ~aligned_markers_data.columns.duplicated()]
    
    #============================================================================
    # 9. RECENTER MARKER COORDINATES RELATIVE TO RCA
    #============================================================================
    rca_columns = {}
    for axis in AXIS_ORDER:
        plain = f"{ZERO_MARKER}_{axis}"
        prefixed = f"Sathy_{ZERO_MARKER}_{axis}"
        if plain in aligned_markers_data.columns:
            rca_columns[axis] = plain
        elif prefixed in aligned_markers_data.columns:
            rca_columns[axis] = prefixed
        else:
            raise ValueError(f"Column for RCA axis {axis} not found in marker data.")
    
    for axis in AXIS_ORDER:
        rca_col = rca_columns[axis]
        rca_series = aligned_markers_data[rca_col]
        if isinstance(rca_series, pd.DataFrame):
            rca_series = rca_series.iloc[:, 0]
    
        axis_cols = [col for col in aligned_markers_data.columns
                     if col.endswith(f"_{axis}")
                     and col not in ["Time (Seconds)", "Frame", rca_col]]
        for col in axis_cols:
            aligned_markers_data.loc[:, col] = aligned_markers_data[col] - rca_series
    
    #============================================================================
    # X. RESAMPLE MARKER DATA TO 2000 Hz
    #============================================================================
    time_array = aligned_markers_data["Time (Seconds)"].values
    orig_fs = 1 / np.median(np.diff(time_array))
    target_fs = 2000.0
    ratio = target_fs / orig_fs
    ratio_frac = Fraction(ratio).limit_denominator(1000)
    p = ratio_frac.numerator
    q = ratio_frac.denominator
    
    marker_cols = [col for col in aligned_markers_data.columns if col not in ["Time (Seconds)", "Frame"]]
    marker_data = aligned_markers_data[marker_cols].values
    
    resampled_data = signal.resample_poly(marker_data, up=p, down=q, axis=0)
    
    new_len = resampled_data.shape[0]
    new_time = np.linspace(time_array[0], time_array[-1], new_len)
    
    resampled_df = pd.DataFrame(resampled_data, columns=marker_cols)
    resampled_df.insert(0, "Time (Seconds)", new_time)
    resampled_df.insert(0, "Frame", range(len(resampled_df)))
    
    aligned_markers_data = resampled_df
    
    #----------------------------------------------------------------------------
    # Remove specified columns and prefixes from marker data
    #----------------------------------------------------------------------------
    for col in ["Active 256_X", "Active 256_Y", "Active 256_Z"]:
        if col in aligned_markers_data.columns:
            aligned_markers_data.drop(columns=[col], inplace=True)
    
    new_columns = {col: col.replace("Sathy_", "") for col in aligned_markers_data.columns if "Sathy_" in col}
    aligned_markers_data.rename(columns=new_columns, inplace=True)
    
    #============================================================================
    # 10. APPLY STANDARD SCALER, SAVE AND VISUALIZE ALIGNED, RECENTERED & RESAMPLED MARKER DATA
    #============================================================================
    # Apply StandardScaler to marker columns (excluding "Frame" and "Time (Seconds)")
    marker_columns = [col for col in aligned_markers_data.columns if col not in ["Frame", "Time (Seconds)"]]
    scaler_markers = StandardScaler()
    aligned_markers_data[marker_columns] = scaler_markers.fit_transform(aligned_markers_data[marker_columns])
    
    aligned_markers_data.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_markers_data_cleaned.csv', index=False)
    print("Aligned, recentered, and resampled marker data saved.")
    
    #for col in aligned_markers_data.columns:
        #if col not in ["Time (Seconds)", "Frame"]:
            #plt.figure(figsize=(12, 6))
            #plt.plot(aligned_markers_data["Time (Seconds)"], aligned_markers_data[col], label=col)
            #plt.axvline(trigger_start, color='green', linestyle='--', label='Trigger Start')
            #plt.axvline(trigger_end, color='red', linestyle='--', label='Trigger End')
            #plt.title(f'Aligned & Recentered & Resampled Marker Data ({col})')
            #plt.xlabel('Time (s)')
            #plt.ylabel('Position (relative to RCA)')
            #plt.legend()
            #plt.grid()
            #plt.show()
    
    
    #============================================================================
    # 12. FUNCTION TO ALIGN AND TRIM EMG DATA BASED ON TRIGGER
    #============================================================================
    def align_emg_data(file_path, trigger_start, trigger_end):
        emg_data = sio.loadmat(file_path)
    
        # Extract Data and Time arrays
        signal_data = np.array(emg_data['Data'][0, 0])
        time_data = np.array(emg_data['Time'][0, 0]).squeeze()
    
        # Create boolean indices for alignment
        aligned_indices = (time_data >= trigger_start) & (time_data <= trigger_end)
        aligned_time = time_data[aligned_indices]
    
        # Determine how to index signal_data based on its dimensions
        if signal_data.ndim == 1:
            aligned_signal = signal_data[aligned_indices]
        elif signal_data.ndim == 2:
            # If the second dimension matches time length, index along axis 1
            if signal_data.shape[1] == time_data.shape[0]:
                aligned_signal = signal_data[:, aligned_indices]
            # If the first dimension matches time length, index along axis 0
            elif signal_data.shape[0] == time_data.shape[0]:
                aligned_signal = signal_data[aligned_indices, :]
            else:
                raise ValueError("Signal data dimensions do not match time data length.")
        else:
            raise ValueError("Signal data has unexpected number of dimensions.")
    
        # Ensure aligned_signal is 2D for consistent processing
        if aligned_signal.ndim == 1:
            aligned_signal = aligned_signal.reshape(-1, 1)
    
        return aligned_time, aligned_signal
    
    #============================================================================
    # 13. PROCESS, VISUALIZE, AND SAVE ALL EMG FILES WITH ICA, INTERPOLATION & CSVs
    #============================================================================
    combined_emg_dfs = []
    # Use marker time as reference for interpolation
    ref_time = aligned_markers_data["Time (Seconds)"].values
    
    for i, emg_file in enumerate(emg_files):
        aligned_time, aligned_signal = align_emg_data(emg_file, trigger_start, trigger_end)
    
        # ------------------------------------------------------------------------
        # Apply StandardScaler before ICA to improve the ICA results
        scaler_before_ica = StandardScaler()
        aligned_signal = scaler_before_ica.fit_transform(aligned_signal)
        # ------------------------------------------------------------------------
    
        # Identify completely constant channels
        channel_variances = np.var(aligned_signal, axis=0)
        non_constant_channels = [ch for ch in range(aligned_signal.shape[1]) if channel_variances[ch] > 1e-12]
        constant_channels = [ch for ch in range(aligned_signal.shape[1]) if ch not in non_constant_channels]
    
        # Apply ICA on non-constant channels
        if len(non_constant_channels) > 0:
            try:
                ica = FastICA(n_components=len(non_constant_channels), random_state=0)
                sources = ica.fit_transform(aligned_signal[:, non_constant_channels])
                reconstructed_non_constant = ica.inverse_transform(sources)
                # Initialize reconstruction array
                reconstructed_signal = np.zeros_like(aligned_signal)
                # Fill reconstructed signals for non-constant channels
                for idx, ch in enumerate(non_constant_channels):
                    reconstructed_signal[:, ch] = reconstructed_non_constant[:, idx]
                # For constant channels, fill with average of non-constant reconstructed signals
                if len(constant_channels) > 0:
                    avg_signal = np.mean(reconstructed_non_constant, axis=1)
                    for ch in constant_channels:
                        reconstructed_signal[:, ch] = avg_signal
                aligned_signal = reconstructed_signal
    
                # ----------------------------------------------------------------
                # Optionally reapply StandardScaler after ICA reconstruction
                scaler_post = StandardScaler()
                aligned_signal = scaler_post.fit_transform(aligned_signal)
                # ----------------------------------------------------------------
    
                print(f"ICA applied to {emg_file} successfully.")
            except Exception as e:
                print(f"ICA failed for {emg_file} with error: {e}. Proceeding without ICA.")
        else:
            print(f"No non-constant channels for ICA in {emg_file}. Skipping ICA.")
    
        # Ensure no NaNs or Infs after ICA
        aligned_signal = np.nan_to_num(aligned_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
        channel_count = aligned_signal.shape[1]
        # Interpolate EMG signals onto marker time grid
        interp_emg = np.zeros((len(ref_time), channel_count))
        for ch in range(channel_count):
            interp_emg[:, ch] = np.interp(ref_time, aligned_time, aligned_signal[:, ch])
    
        new_len = len(ref_time)
        new_time = ref_time.copy()
    
        # Prepare DataFrame for current EMG file channels
        emg_df_dict = {
             "Frame": range(new_len),
             "Time (Seconds)": new_time,
        }
        for ch in range(channel_count):
            emg_df_dict[f"EMG_Channel_{i+1}_Ch{ch+1}"] = interp_emg[:, ch]
        emg_df = pd.DataFrame(emg_df_dict)
    
        # Save separate CSV for each EMG file containing its channels
        emg_csv_path =f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i+1}.csv"
        emg_df.to_csv(emg_csv_path, index=False)
        print(f"Aligned EMG CSV saved to {emg_csv_path}")
    
        combined_emg_dfs.append(emg_df)
    
        # Save aligned EMG data as .mat for the entire file (all channels)
        aligned_emg_data = {
            'Aligned_Time': new_time,
            'Aligned_Signal': interp_emg
        }
        save_path = f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i + 1}.mat"
        sio.savemat(save_path, aligned_emg_data)
        print(f"Aligned EMG saved to {save_path}")
    
    #----------------------------------------------------------------------------
    # Combine all EMG files' channels into a single CSV
    #----------------------------------------------------------------------------
    #if combined_emg_dfs:
        #combined_df = combined_emg_dfs[0][["Frame", "Time (Seconds)"]].copy()
        #for df in combined_emg_dfs:
            #for col in df.columns:
                #if col not in ["Frame", "Time (Seconds)"]:
                    #combined_df[col] = df[col]
        #combined_csv_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\Gait_Phase\\Data\\combined_EMG_data.csv'
        #combined_df.to_csv(combined_csv_path, index=False)
        #print(f"Combined EMG CSV saved to {combined_csv_path}")

    if combined_emg_dfs:
        combined_df = combined_emg_dfs[0][["Frame", "Time (Seconds)"]].copy()
        data_dfs = []
        for df in combined_emg_dfs:
            cols_to_use = [col for col in df.columns if col not in ["Frame", "Time (Seconds)"]]
            if cols_to_use:
                data_dfs.append(df[cols_to_use])
        if data_dfs:
            data_combined = pd.concat(data_dfs, axis=1)
            combined_df = pd.concat([combined_df, data_combined], axis=1)
        combined_csv_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\combined_EMG_data.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined EMG CSV saved to {combined_csv_path}")
    # -------------------------------------------------------------------------
    # 14. READ YOUR CSV AND PREPARE FUNCTIONS FOR FEATURE CALCULATION
    # -------------------------------------------------------------------------
    input_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\combined_EMG_data.csv'
    df = pd.read_csv(input_path)
    
    # Identify columns
    frame_col = 'Frame'
    time_col = 'Time (Seconds)'
    channel_cols = [col for col in df.columns if col.startswith('EMG_Channel')]
    
    # -------------------------------------------------------------------------
    # 15. APPLY BANDPASS FILTERING TO EACH EMG CHANNEL
    # -------------------------------------------------------------------------
    # Define bandpass filter parameters
    lowcut = 20.0    # Hz
    highcut = 450.0  # Hz
    order = 4
    
    # Compute sampling frequency dynamically from time column
    time_values = df[time_col].values
    fs = 1 / np.mean(np.diff(time_values))
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply bandpass filter to each channel
    for ch in channel_cols:
        df[ch] = sosfilt(sos, df[ch].values)
    
    # -------------------------------------------------------------------------
    # 16. COMPUTE WINDOW SIZE DYNAMICALLY BASED ON SAMPLING RATE
    # -------------------------------------------------------------------------
    desired_window_sec = 0.1
    window_size = int(fs * desired_window_sec)
    if window_size < 1:
        window_size = 1
    
    # -------------------------------------------------------------------------
    # 17. USE ROLLING WINDOW OPERATIONS FOR EFFICIENT FEATURE COMPUTATION
    # -------------------------------------------------------------------------
    features = {frame_col: df[frame_col], time_col: df[time_col]}
    
    for ch in tqdm(channel_cols, desc="Processing channels"):
        series = df[ch]
        abs_series = series.abs()
        squared_series = series.pow(2)
        abs_diff = series.diff().abs()
    
        # Compute rolling features
        WL = abs_diff.rolling(window=window_size, min_periods=1).sum()
        MAV = abs_series.rolling(window=window_size, min_periods=1).mean()
        IAV = abs_series.rolling(window=window_size, min_periods=1).sum()
        RMS = squared_series.rolling(window=window_size, min_periods=1).mean().pow(0.5)
        # Avoid division by zero for AAC calculation
        AAC = WL / (window_size - 1) if window_size > 1 else WL
    
        # Store features in the dictionary
        features[f'{ch}_WL'] = WL
        features[f'{ch}_MAV'] = MAV
        features[f'{ch}_IAV'] = IAV
        features[f'{ch}_RMS'] = RMS
        features[f'{ch}_AAC'] = AAC
    
    # -------------------------------------------------------------------------
    # 18. CONVERT FEATURES DICTIONARY TO DATAFRAME
    # -------------------------------------------------------------------------
    features_df = pd.DataFrame(features)
    
    # -------------------------------------------------------------------------
    # 19. APPLY MIN-MAX SCALER TO FEATURE COLUMNS
    # -------------------------------------------------------------------------
    scaler = MinMaxScaler()
    feature_columns = [col for col in features_df.columns if col not in [frame_col, time_col]]
    features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])
    
    # -------------------------------------------------------------------------
    # 20. COUNT NaN AND ZERO VALUES FOR EACH PROBE BEFORE INTERPOLATION
    # -------------------------------------------------------------------------
    probes = {1: [], 2: [], 3: [], 4: []}
    
    for col in features_df.columns:
        if col.startswith("EMG_Channel_1"):
            probes[1].append(col)
        elif col.startswith("EMG_Channel_2"):
            probes[2].append(col)
        elif col.startswith("EMG_Channel_3"):
            probes[3].append(col)
        elif col.startswith("EMG_Channel_4"):
            probes[4].append(col)
    
    nan_counts_before = {}
    zero_counts_before = {}
    
    for probe, cols in probes.items():
        nan_counts_before[probe] = features_df[cols].isna().sum().sum()
        zero_counts_before[probe] = (features_df[cols] == 0).sum().sum()
    
    # -------------------------------------------------------------------------
    # 21. APPLY LINEAR INTERPOLATION TO FILL NaN AND ZERO VALUES
    # -------------------------------------------------------------------------
    for col in features_df.columns:
        if col in [frame_col, time_col]:
            continue
        # Convert zeros to NaNs for interpolation, then interpolate
        features_df[col] = features_df[col].replace(0, np.nan)
        features_df[col] = features_df[col].interpolate(method='linear', limit_direction='both')
    
    # -------------------------------------------------------------------------
    # 22. COUNT NaN AND ZERO VALUES AFTER INTERPOLATION
    # -------------------------------------------------------------------------
    nan_counts_after = {}
    zero_counts_after = {}
    
    for probe, cols in probes.items():
        nan_counts_after[probe] = features_df[cols].isna().sum().sum()
        zero_counts_after[probe] = (features_df[cols] == 0).sum().sum()
    
    # -------------------------------------------------------------------------
    # 23. COUNT NEGATIVE VALUES BEFORE ABSOLUTE VALUE TRANSFORMATION
    # -------------------------------------------------------------------------
    neg_counts_before = {}
    for probe, cols in probes.items():
        neg_counts_before[probe] = (features_df[cols] < 0).sum().sum()
    
    print("\n=== Negative Value Summary BEFORE ABSOLUTE VALUE TRANSFORMATION ===\n")
    for probe in probes:
        print(f"Probe {probe}: {neg_counts_before[probe]} negative values")
    
    # -------------------------------------------------------------------------
    # 24. APPLY ABSOLUTE VALUE TRANSFORMATION TO ENSURE ALL FEATURE VALUES ARE POSITIVE
    # -------------------------------------------------------------------------
    features_df[feature_columns] = features_df[feature_columns].abs()
    
    # -------------------------------------------------------------------------
    # 25. COUNT NEGATIVE VALUES AFTER ABSOLUTE VALUE TRANSFORMATION
    # -------------------------------------------------------------------------
    neg_counts_after = {}
    for probe, cols in probes.items():
        neg_counts_after[probe] = (features_df[cols] < 0).sum().sum()
    
    print("\n=== Negative Value Summary AFTER ABSOLUTE VALUE TRANSFORMATION ===\n")
    for probe in probes:
        print(f"Probe {probe}: {neg_counts_after[probe]} negative values")
    
    # -------------------------------------------------------------------------
    # 26. SAVE FEATURES DATAFRAME TO CSV
    # -------------------------------------------------------------------------
    output_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\emg_all_samples_features.csv"
    features_df.to_csv(output_path, index=False)
    
    # -------------------------------------------------------------------------
    # 27. PRINT NaN and ZERO VALUE SUMMARY
    # -------------------------------------------------------------------------
    print("\n=== NaN and Zero Value Summary ===\n")
    for probe in probes:
        print(f"Probe {probe}:")
        print(f"  NaN values before interpolation: {nan_counts_before[probe]}")
        print(f"  Zero values before interpolation: {zero_counts_before[probe]}")
        print(f"  NaN values after interpolation: {nan_counts_after[probe]}")
        print(f"  Zero values after interpolation: {zero_counts_after[probe]}\n")
    
    print(f"Per-sample channel features saved to: {output_path}")

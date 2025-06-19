import csv
import numpy as np
import pandas as pd
import ezc3d
import re
import scipy.io as sio
from Gait_Phase.gpu_utils import configure_gpu
from fractions import Fraction
from scipy import signal
from scipy.signal import sosfilt, butter
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

configure_gpu()

def align_emg_data_pipeline():
    # -------------------------------------------------------------------------
    # 1. LOAD C3D FILE AND EXPORT MARKER / FORCE-PLATE DATA
    # -------------------------------------------------------------------------
    c3d = ezc3d.c3d('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Trial02.c3d')

    # --- 1) Export marker trajectories to CSV ---
    points      = c3d['data']['points']
    labels      = c3d['parameters']['POINT']['LABELS']['value']
    n_frames    = points.shape[2]
    marker_rate = float(c3d['parameters']['POINT']['RATE']['value'][0])
    marker_time = np.arange(n_frames) / marker_rate

    marker_data = []
    for f in range(n_frames):
        row = {
            'Frame':  f + 1,
            'Time_s': float(f"{marker_time[f]:.3f}")
        }
        for m, lbl in enumerate(labels):
            x, y, z, _ = points[:, m, f]
            row[f"{lbl}_X"] = x
            row[f"{lbl}_Y"] = y
            row[f"{lbl}_Z"] = z
        marker_data.append(row)

    markers_df = pd.DataFrame(marker_data)
    markers_df.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\markers.csv', index=False)

    # --- 2) Export raw analog channels to CSV ---
    analog_raw    = c3d['data']['analogs']
    subs, n_ch, n_fr = analog_raw.shape
    flat          = analog_raw.transpose(2, 0, 1).reshape(n_fr * subs, n_ch)
    analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']
    pd.DataFrame(flat, columns=analog_labels) \
        .to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\analogs.csv', index=False)

    # --- 3) Build force-plate CSV with COP ---
    rate       = float(c3d['parameters']['ANALOG']['RATE']['value'][0])
    time       = np.arange(n_fr * subs) / rate
    plate_nums = sorted({
        int(re.search(r'Force\.Fx(\d+)', lbl).group(1))
        for lbl in analog_labels if 'Force.Fx' in lbl
    })

    force_dfs = []
    for p in plate_nums:
        Fx, Fy, Fz = f'Force.Fx{p}', f'Force.Fy{p}', f'Force.Fz{p}'
        Mx, My     = f'Moment.Mx{p}', f'Moment.My{p}'
        dfp = pd.DataFrame({
            'Time_s': time,
            f'F{p}_X': flat[:, analog_labels.index(Fx)],
            f'F{p}_Y': flat[:, analog_labels.index(Fy)],
            f'F{p}_Z': flat[:, analog_labels.index(Fz)],
            f'M{p}_X': flat[:, analog_labels.index(Mx)],
            f'M{p}_Y': flat[:, analog_labels.index(My)],
        })
        # COP
        dfp[f'COP{p}_X'] = -dfp[f'M{p}_Y'] / dfp[f'F{p}_Z']
        dfp[f'COP{p}_Y'] =  dfp[f'M{p}_X'] / dfp[f'F{p}_Z']
        force_dfs.append(dfp)

    force_df = pd.concat(force_dfs, axis=1)
    force_df.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\force_plates.csv', index=False)

    # -------------------------------------------------------------------------
    # 2. HEEL-STRIKE DETECTION AND FILTERING
    # -------------------------------------------------------------------------
    F2_Z      = -force_df['F2_Z']
    mask      = (F2_Z.shift(1) < 5) & (F2_Z >= 5)
    hs_times  = time[mask.values]                # strike times (seconds)
    hs_frames = (hs_times * rate).astype(int)    # analog-rate frames

    # Filter strikes by minimum stride duration (≥ 0.7 s)
    min_dur           = 0.7
    filtered_times    = [hs_times[0]]
    filtered_frames   = [hs_frames[0]]
    for t, fr in zip(hs_times[1:], hs_frames[1:]):
        if (t - filtered_times[-1]) >= min_dur:
            filtered_times.append(t)
            filtered_frames.append(fr)

    pd.DataFrame({
        'Heel_Strike_Time_s': filtered_times,
        'Heel_Strike_Frame':  filtered_frames
    }).to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strikes.csv', index=False)

    # -------------------------------------------------------------------------
    # 3. RESAMPLE MARKERS TO 1000 Hz AND EXTRACT STRIKES
    # -------------------------------------------------------------------------
    fs_new    = 1000
    t0, t1    = markers_df['Time_s'].iloc[0], markers_df['Time_s'].iloc[-1]
    new_times = np.arange(t0, t1, 1/fs_new)
    n_new     = new_times.size

    resampled = {
        'Frame':  np.arange(1, n_new + 1),
        'Time_s': new_times
    }
    for lbl in labels:
        for ax in ('X', 'Y', 'Z'):
            col = f"{lbl}_{ax}"
            resampled[col] = np.interp(new_times, markers_df['Time_s'], markers_df[col])

    resampled_df = pd.DataFrame(resampled)
    resampled_df.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\markers_resampled_1000Hz.csv', index=False)

    # Extract marker data at filtered strikes only
    filtered_marker_frames = (np.array(filtered_times) * fs_new).astype(int)
    hs_markers_filtered = resampled_df.iloc[filtered_marker_frames].copy()
    hs_markers_filtered.insert(0, 'Heel_Strike_Time_s', filtered_times)
    hs_markers_filtered.insert(0, 'Heel_Strike_Marker_Frame', filtered_marker_frames)
    hs_markers_filtered.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strike_markers.csv', index=False)

    print("Saved:\n"
          " • C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strikes.csv\n"
          " • C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\markers_resampled_1000Hz.csv\n"
          " • C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strike_markers.csv")

    # -------------------------------------------------------------------------
    # 4. ALIGN MARKERS TO TRIGGER & RESAMPLE TO 2000 Hz (PLUS RECENTERING)
    # -------------------------------------------------------------------------
    marker_file_path  = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\markers_resampled_1000Hz.csv'
    trigger_file_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\trigger_signal.mat'
    emg_files = [
        'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi1_EMG.mat',
        #'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi2_EMG.mat',
        'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi3_EMG.mat',
        'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\Muovi4_EMG.mat',
    ]

    marker_headers_row   = 0
    data_start_row       = 1
    AXIS_ORDER           = ["X", "Y", "Z"]
    INTERPOLATION_METHOD = "linear"
    SKIP_UNLABELED       = True
    ZERO_MARKER          = "RKNE"

    # -- Read raw CSV ---------------------------------------------------------
    with open(marker_file_path, "r", newline="") as f:
        reader = csv.reader(f)
        lines  = list(reader)

    # -- Extract header & marker labels --------------------------------------
    raw_header       = lines[marker_headers_row]
    raw_marker_labels = raw_header[2:]        # skip Frame + Time_s
    num_raw_cols     = len(raw_marker_labels)
    num_groups       = num_raw_cols // 3

    # Helper to strip axis suffix
    def strip_final_axis(label):
        for ax in ["_X", "_Y", "_Z"]:
            if label.endswith(ax):
                return label[:-2]
        return label

    marker_names   = []
    group_indices  = []
    for g in range(num_groups):
        cX, cY, cZ = 3*g, 3*g+1, 3*g+2
        lx, ly, lz = raw_marker_labels[cX].strip(), raw_marker_labels[cY].strip(), raw_marker_labels[cZ].strip()
        nx, ny, nz = strip_final_axis(lx), strip_final_axis(ly), strip_final_axis(lz)
        if nx == ny == nz:
            mname = nx
        else:
            raise ValueError(f"Inconsistent marker names: {lx}, {ly}, {lz}")
        if SKIP_UNLABELED and "Unlabeled" in mname:
            continue
        marker_names.append(mname)
        group_indices.append((cX, cY, cZ))

    # Build DataFrame headers
    final_headers = [f"{m}_{ax}" for m in marker_names for ax in AXIS_ORDER]

    # -- Extract numeric data & time -----------------------------------------
    data_rows = lines[data_start_row:]
    frame_col = [int(r[0])   for r in data_rows]
    time_col  = [float(r[1]) for r in data_rows]

    numeric = []
    for row in data_rows:
        slice_ = row[2:]
        vals   = []
        for ix, iy, iz in group_indices:
            def to_f(v):
                try:
                    return float(v) if v != "" else np.nan
                except:
                    return np.nan
            vals.extend([to_f(slice_[ix]), to_f(slice_[iy]), to_f(slice_[iz])])
        numeric.append(vals)

    df = pd.DataFrame(numeric, columns=final_headers)
    df.insert(0, "Frame", frame_col)
    df.insert(1, "Time_s", time_col)

    # -- Interpolate missing --------------------------------------------------
    df = df.interpolate(method=INTERPOLATION_METHOD,
                        limit_direction="both",
                        axis=0).round(6)
    print(f"Marker DF shape after interpolation: {df.shape}")

    # -- Load trigger signal --------------------------------------------------
    trig            = sio.loadmat(trigger_file_path)
    trigger_time    = trig['Time'][0, 0].squeeze()
    trigger_signal  = trig['Data'][0, 0].squeeze()
    trigger_start   = trigger_time[0]
    trigger_end     = trigger_time[-1]
    print(f"Trigger start: {trigger_start}, end: {trigger_end}")

    # -- Align & trim markers -------------------------------------------------
    aligned_markers = df[(df["Time_s"] >= trigger_start) &
                         (df["Time_s"] <= trigger_end)].copy()
    aligned_markers = aligned_markers.loc[:, ~aligned_markers.columns.duplicated()]

    # -- Recenter to RKNE -----------------------------------------------------
    for ax in AXIS_ORDER:
        base_col = f"{ZERO_MARKER}_{ax}"
        if base_col not in aligned_markers:
            raise ValueError(f"Missing recenter column: {base_col}")
        base = aligned_markers[base_col]
        for col in [c for c in aligned_markers if c.endswith(f"_{ax}") and c != base_col]:
            aligned_markers[col] -= base

    # -- Resample to 2000 Hz --------------------------------------------------
    t_arr   = aligned_markers["Time_s"].values
    fs_orig = 1 / np.median(np.diff(t_arr))
    up, down = Fraction(2000.0 / fs_orig).limit_denominator(1000).as_integer_ratio()

    marker_cols = [c for c in aligned_markers if c not in ["Frame", "Time_s"]]
    res         = signal.resample_poly(aligned_markers[marker_cols].values,
                                       up=up,
                                       down=down,
                                       axis=0)
    new_len   = res.shape[0]
    new_t     = np.linspace(t_arr[0], t_arr[-1], new_len)
    res_df    = pd.DataFrame(res, columns=marker_cols)
    res_df.insert(0, "Frame", range(new_len))
    res_df.insert(1, "Time_s", new_t)
    aligned_markers = res_df

    # -- Scale and save -------------------------------------------------------
    scaler_m = StandardScaler()
    cols_to_scale = [c for c in aligned_markers if c not in ["Frame", "Time_s"]]
    aligned_markers[cols_to_scale] = scaler_m.fit_transform(aligned_markers[cols_to_scale])
    aligned_markers.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_markers_data_cleaned.csv', index=False)
    print("Aligned markers saved.")

    # -- Extract filtered heel-strike markers at 2000 Hz ----------------------
    fs_marker           = 2000
    marker_idx_2000     = (np.array(filtered_times) * fs_marker).astype(int)
    marker_idx_2000     = np.clip(marker_idx_2000, 0, aligned_markers.shape[0] - 1)
    hs2000              = aligned_markers.iloc[marker_idx_2000].copy()
    hs2000.insert(0, 'Heel_Strike_Time_s', filtered_times)
    hs2000.insert(0, 'Heel_Strike_Frame_2000Hz', marker_idx_2000)
    hs2000.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strike_markers_2000Hz.csv', index=False)
    print("→ Saved filtered heel-strike markers at 2000 Hz to C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\filtered_heel_strike_markers_2000Hz.csv")

    # -------------------------------------------------------------------------
    # 5. EMG PROCESSING (FIRST PASS – IDENTICAL TO ORIGINAL)
    # -------------------------------------------------------------------------
    def align_emg_data(file_path, start, end):
        mat = sio.loadmat(file_path)
        sig = np.array(mat['Data'][0, 0])
        t   = np.array(mat['Time'][0, 0]).squeeze()
        idx = (t >= start) & (t <= end)
        at  = t[idx]
        if sig.ndim == 1:
            asig = sig[idx]
        elif sig.ndim == 2:
            if sig.shape[1] == t.shape[0]:
                asig = sig[:, idx]
            elif sig.shape[0] == t.shape[0]:
                asig = sig[idx, :]
            else:
                raise ValueError("Signal/time mismatch")
        if asig.ndim == 1:
            asig = asig.reshape(-1, 1)
        return at, asig

    combined_emg = []
    ref_time = aligned_markers["Time_s"].values

    for i, ef in enumerate(emg_files):
        at, asig = align_emg_data(ef, trigger_start, trigger_end)

        # Standardize before ICA
        asig = StandardScaler().fit_transform(asig)

        # ICA
        vars_ch  = np.var(asig, axis=0)
        non_const = [ch for ch in range(asig.shape[1]) if vars_ch[ch] > 1e-12]
        const_ch  = [ch for ch in range(asig.shape[1]) if ch not in non_const]

        if non_const:
            try:
                ica  = FastICA(n_components=len(non_const), random_state=0)
                src  = ica.fit_transform(asig[:, non_const])
                rec  = ica.inverse_transform(src)
                recon = np.zeros_like(asig)
                for idx, ch in enumerate(non_const):
                    recon[:, ch] = rec[:, idx]
                if const_ch:
                    avg = np.mean(rec, axis=1)
                    for ch in const_ch:
                        recon[:, ch] = avg
                asig = StandardScaler().fit_transform(recon)
                print(f"ICA applied on {ef}")
            except Exception as e:
                print(f"ICA failed for {ef}: {e}")
        else:
            print(f"No ICA channels for {ef}")

        asig = np.nan_to_num(asig)

        # Interpolate to marker time grid
        interp = np.zeros((len(ref_time), asig.shape[1]))
        for ch in range(asig.shape[1]):
            interp[:, ch] = np.interp(ref_time, at, asig[:, ch])

        emg_df = pd.DataFrame({
            "Frame":  range(len(ref_time)),
            "Time_s": ref_time
        })
        for ch in range(asig.shape[1]):
            emg_df[f"EMG_File{i+1}_Ch{ch+1}"] = interp[:, ch]

        emg_df.to_csv(f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i+1}.csv", index=False)
        sio.savemat(f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i+1}.mat", {
            "Aligned_Time": ref_time,
            "Aligned_Signal": interp
        })
        combined_emg.append(emg_df)

    # Combine all EMG (first-pass)
    if combined_emg:
        # Start with Frame and Time_s
        comb = combined_emg[0][["Frame", "Time_s"]].copy()

        # Collect all dataframes to concatenate
        dfs_to_concat = [comb]

        for df_em in combined_emg:
            new_cols = df_em.drop(columns=["Frame", "Time_s"])
            dfs_to_concat.append(new_cols)

        # Concatenate all at once along columns
        comb = pd.concat(dfs_to_concat, axis=1)
        comb.to_csv('C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\combined_EMG_data.csv', index=False)
        print("Combined EMG saved.")

    # -------------------------------------------------------------------------
    # 6. EMG PROCESSING (FULL FEATURE PIPELINE – SECOND PASS)
    # -------------------------------------------------------------------------
    combined_emg_dfs = []
    aligned_markers_data = aligned_markers  # alias to match later code
    ref_time = aligned_markers_data["Time_s"].values

    for i, emg_file in enumerate(emg_files):
        aligned_time, aligned_signal = align_emg_data(emg_file, trigger_start, trigger_end)

        # Standardize before ICA
        scaler_before_ica = StandardScaler()
        aligned_signal = scaler_before_ica.fit_transform(aligned_signal)

        # Identify completely constant channels
        channel_variances     = np.var(aligned_signal, axis=0)
        non_constant_channels = [ch for ch in range(aligned_signal.shape[1]) if channel_variances[ch] > 1e-12]
        constant_channels     = [ch for ch in range(aligned_signal.shape[1]) if ch not in non_constant_channels]

        # Apply ICA on non-constant channels
        if len(non_constant_channels) > 0:
            try:
                ica = FastICA(n_components=len(non_constant_channels), random_state=0)
                sources = ica.fit_transform(aligned_signal[:, non_constant_channels])
                reconstructed_non_constant = ica.inverse_transform(sources)

                # Reconstruct full matrix
                reconstructed_signal = np.zeros_like(aligned_signal)
                for idx, ch in enumerate(non_constant_channels):
                    reconstructed_signal[:, ch] = reconstructed_non_constant[:, idx]

                # Fill constant channels
                if len(constant_channels) > 0:
                    avg_signal = np.mean(reconstructed_non_constant, axis=1)
                    for ch in constant_channels:
                        reconstructed_signal[:, ch] = avg_signal

                aligned_signal = reconstructed_signal

                # re-standardize
                scaler_post = StandardScaler()
                aligned_signal = scaler_post.fit_transform(aligned_signal)

                print(f"ICA applied to {emg_file} successfully.")
            except Exception as e:
                print(f"ICA failed for {emg_file} with error: {e}. Proceeding without ICA.")
        else:
            print(f"No non-constant channels for ICA in {emg_file}. Skipping ICA.")

        # Ensured finite values
        aligned_signal = np.nan_to_num(aligned_signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Interpolate onto marker grid
        channel_count = aligned_signal.shape[1]
        interp_emg    = np.zeros((len(ref_time), channel_count))
        for ch in range(channel_count):
            interp_emg[:, ch] = np.interp(ref_time, aligned_time, aligned_signal[:, ch])

        new_len  = len(ref_time)
        new_time = ref_time.copy()

        # Prepare DataFrame
        emg_df_dict = {
            "Frame": range(new_len),
            "Time (Seconds)": new_time,
        }
        for ch in range(channel_count):
            emg_df_dict[f"EMG_Channel_{i+1}_Ch{ch+1}"] = interp_emg[:, ch]

        emg_df = pd.DataFrame(emg_df_dict)

        emg_csv_path = f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i+1}.csv"
        emg_df.to_csv(emg_csv_path, index=False)
        print(f"Aligned EMG CSV saved to {emg_csv_path}")

        combined_emg_dfs.append(emg_df)

        # Save MAT
        aligned_emg_data = {
            'Aligned_Time': new_time,
            'Aligned_Signal': interp_emg
        }
        save_path = f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_EMG_File_{i + 1}.mat"
        sio.savemat(save_path, aligned_emg_data)
        print(f"Aligned EMG saved to {save_path}")

    # -- Combine all EMG channels into one CSV --------------------------------
    if combined_emg_dfs:
        combined_df = combined_emg_dfs[0][["Frame", "Time (Seconds)"]].copy()
        data_dfs    = []

        for df in combined_emg_dfs:
            cols_to_use = [col for col in df.columns if col not in ["Frame", "Time (Seconds)"]]
            if cols_to_use:
                data_dfs.append(df[cols_to_use])

        if data_dfs:
            data_combined = pd.concat(data_dfs, axis=1)
            combined_df   = pd.concat([combined_df, data_combined], axis=1)

        combined_csv_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\combined_EMG_data.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined EMG CSV saved to {combined_csv_path}")

    # -------------------------------------------------------------------------
    # 7. FEATURE EXTRACTION (WINDOWED PER-SAMPLE METRICS)
    # -------------------------------------------------------------------------
    input_path = 'C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\combined_EMG_data.csv'
    df         = pd.read_csv(input_path)

    frame_col      = 'Frame'
    time_col       = 'Time (Seconds)'
    channel_cols   = [col for col in df.columns if col.startswith('EMG_Channel')]

    # Band-pass filter
    lowcut, highcut, order = 20.0, 450.0, 4
    time_values   = df[time_col].values
    fs            = 1 / np.mean(np.diff(time_values))
    nyq           = 0.5 * fs
    low, high     = lowcut / nyq, highcut / nyq
    sos           = butter(order, [low, high], btype='band', output='sos')

    for ch in channel_cols:
        df[ch] = sosfilt(sos, df[ch].values)

    # Window size (0.1 s)
    desired_window_sec = 0.1
    window_size        = int(fs * desired_window_sec)
    window_size        = max(window_size, 1)

    # Rolling features
    features = {frame_col: df[frame_col], time_col: df[time_col]}

    for ch in tqdm(channel_cols, desc="Processing channels"):
        series         = df[ch]
        abs_series     = series.abs()
        squared_series = series.pow(2)
        abs_diff       = series.diff().abs()

        WL  = abs_diff.rolling(window=window_size, min_periods=1).sum()
        MAV = abs_series.rolling(window=window_size, min_periods=1).mean()
        IAV = abs_series.rolling(window=window_size, min_periods=1).sum()
        RMS = squared_series.rolling(window=window_size, min_periods=1).mean().pow(0.5)
        AAC = WL / (window_size - 1) if window_size > 1 else WL

        features[f'{ch}_WL']  = WL
        features[f'{ch}_MAV'] = MAV
        features[f'{ch}_IAV'] = IAV
        features[f'{ch}_RMS'] = RMS
        features[f'{ch}_AAC'] = AAC

    features_df = pd.DataFrame(features)

    # Min-max scale
    scaler            = MinMaxScaler()
    feature_columns    = [col for col in features_df.columns if col not in [frame_col, time_col]]
    features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])

    # Probe grouping
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

    nan_counts_before, zero_counts_before = {}, {}
    for probe, cols in probes.items():
        nan_counts_before[probe]  = features_df[cols].isna().sum().sum()
        zero_counts_before[probe] = (features_df[cols] == 0).sum().sum()

    # Interpolate NaNs / zeros
    for col in features_df.columns:
        if col in [frame_col, time_col]:
            continue
        features_df[col] = features_df[col].replace(0, np.nan)
        features_df[col] = features_df[col].interpolate(method='linear', limit_direction='both')

    nan_counts_after, zero_counts_after = {}, {}
    for probe, cols in probes.items():
        nan_counts_after[probe]  = features_df[cols].isna().sum().sum()
        zero_counts_after[probe] = (features_df[cols] == 0).sum().sum()

    # Negative value counts
    neg_counts_before = {probe: (features_df[cols] < 0).sum().sum() for probe, cols in probes.items()}
    print("\n=== Negative Value Summary BEFORE ABSOLUTE VALUE TRANSFORMATION ===\n")
    for probe in probes:
        print(f"Probe {probe}: {neg_counts_before[probe]} negative values")

    # Absolute value transform
    features_df[feature_columns] = features_df[feature_columns].abs()

    neg_counts_after = {probe: (features_df[cols] < 0).sum().sum() for probe, cols in probes.items()}
    print("\n=== Negative Value Summary AFTER ABSOLUTE VALUE TRANSFORMATION ===\n")
    for probe in probes:
        print(f"Probe {probe}: {neg_counts_after[probe]} negative values")

    # Save features
    output_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\emg_all_samples_features.csv"
    features_df.to_csv(output_path, index=False)

    # Summary
    print("\n=== NaN and Zero Value Summary ===\n")
    for probe in probes:
        print(f"Probe {probe}:")
        print(f"  NaN values before interpolation: {nan_counts_before[probe]}")
        print(f"  Zero values before interpolation: {zero_counts_before[probe]}")
        print(f"  NaN values after interpolation: {nan_counts_after[probe]}")
        print(f"  Zero values after interpolation: {zero_counts_after[probe]}\n")

    print(f"Per-sample channel features saved to: {output_path}")

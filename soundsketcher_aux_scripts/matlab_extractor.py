import matlab.engine
import numpy as np
import subprocess
import time


def to_array(data):
    arr = np.array(data)
    arr = np.squeeze(arr)
    if arr.ndim > 1 and arr.shape[0] == 2:
        arr = arr[0]
    return arr


def interpolate_series(timestamps, time_vector, value_vector):
    return np.interp(timestamps, time_vector, value_vector)

#! Updated to work with MATLAB worker thread | Disabled old features
def extract_mir_features_for_file_interpolated(
    eng,
    audio_path: str,
    target_timestamps: np.ndarray
) -> dict:
    """
    Extract MIR features from an audio file and interpolate them onto given timestamps.
    Includes roughness, sharpness, loudness, and spectral centroid (weighted and bark2hz).

    Args:
        audio_path (str): Full path to audio file.
        target_timestamps (np.ndarray): Array of time points to interpolate onto.

    Returns:
        dict: Interpolated MIR features + original raw data.
    """

    # Run roughnessTimeSeries
    (
        mps_roughness, rough_z, sharp_z, rough_v, rough_s, Loudness_SC_Hz,
        time_MPS, time_Zwicker, time_v, time_s
    ) = eng.roughnessTimeSeries(audio_path,nargout = 10) #! old features are calculated here

    # # Run SC_Loudness_single for Bark-based spectral centroid
    centroid_bark2hz,time_bark2hz = eng.SC_Loudness_single(audio_path,nargout = 2)

    # Helper to convert MATLAB output to NumPy array
    def unwrap(x): return to_array(x[0]) if isinstance(x, (list, tuple)) else to_array(x)

    features = {
        'MPS_roughness': unwrap(mps_roughness),
        # 'roughness_Zwicker': unwrap(rough_z),
        'sharpness_Zwicker': unwrap(sharp_z),
        'roughness_vassilakis': unwrap(rough_v),
        # 'roughness_sethares': unwrap(rough_s),
        'weighted_spectral_centroid': unwrap(Loudness_SC_Hz),
        # 'weighted_spectral_centroid': unwrap(centroid_bark2hz), # this works
        # 'spectral_centroid_bark2hz': unwrap(centroid_bark2hz),
        'time_MPS': unwrap(time_MPS),
        'time_Zwicker': unwrap(time_Zwicker),
        'time_vassilakis': unwrap(time_v),
        # 'time_sethares': unwrap(time_s),
        'time_bark2hz': unwrap(time_bark2hz),
    }

    # Interpolate features to target timestamps
    interpolated = {
        'MPS_roughness': interpolate_series(target_timestamps, features['time_MPS'], features['MPS_roughness']),
        # 'roughness_Zwicker': interpolate_series(target_timestamps, features['time_Zwicker'], features['roughness_Zwicker']),
        'sharpness_Zwicker': interpolate_series(target_timestamps, features['time_Zwicker'], features['sharpness_Zwicker']),
        'roughness_vassilakis': interpolate_series(target_timestamps, features['time_vassilakis'], features['roughness_vassilakis']),
        # 'roughness_sethares': interpolate_series(target_timestamps, features['time_sethares'], features['roughness_sethares']),
        'weighted_spectral_centroid': interpolate_series(target_timestamps, features['time_Zwicker'], features['weighted_spectral_centroid']),
        # 'weighted_spectral_centroid': interpolate_series(target_timestamps, features['time_bark2hz'], features['weighted_spectral_centroid']), # this works
        # 'spectral_centroid_bark2hz': interpolate_series(target_timestamps, features['time_bark2hz'], features['spectral_centroid_bark2hz']),
        'timestamps': target_timestamps
    }

    return interpolated

#! Previous version 'extract_mir_features_for_file_interpolated'
# def extract_mir_features_for_file_interpolated(
#     audio_path: str,
#     mirtoolbox_path: str,
#     matlab_script_path: str,
#     target_timestamps: np.ndarray
# ) -> dict:
#     """
#     Extract MIR features from an audio file and interpolate them onto given timestamps.
#     Includes roughness, sharpness, loudness, and spectral centroid (weighted and bark2hz).

#     Args:
#         audio_path (str): Full path to audio file.
#         mirtoolbox_path (str): Path to MIRToolbox.
#         matlab_script_path (str): Path where MATLAB scripts are located.
#         target_timestamps (np.ndarray): Array of time points to interpolate onto.

#     Returns:
#         dict: Interpolated MIR features + original raw data.
#     """
#     import matlab.engine

#     # DEBUG: Write file path to a log file
#     debug_log_path = "/mnt/ssd1/kvelenis/soundsketcher/debug_mir_paths.txt"
#     with open(debug_log_path, "a") as f:
#         f.write(f"[MIR] file_path = {audio_path}\n")

#     # Start MATLAB engine and add paths
#     eng = matlab.engine.start_matlab()
#     eng.addpath(eng.genpath(mirtoolbox_path), nargout=0)
#     eng.cd(matlab_script_path)

#     # Run roughnessTimeSeries
#     (
#         mps_roughness, rough_z, sharp_z, rough_v, rough_s, Loudness_SC_Hz,
#         time_MPS, time_Zwicker, time_v, time_s
#     ) = eng.roughnessTimeSeries(audio_path, nargout=10)

#     # Run SC_Loudness_single for Bark-based spectral centroid
#     centroid_bark2hz, time_bark2hz = eng.SC_Loudness_single(audio_path, nargout=2)

#     eng.quit()

#     # Helper to convert MATLAB output to NumPy array
#     def unwrap(x): return to_array(x[0]) if isinstance(x, (list, tuple)) else to_array(x)

#     features = {
#         'MPS_roughness': unwrap(mps_roughness),
#         'roughness_Zwicker': unwrap(rough_z),
#         'sharpness_Zwicker': unwrap(sharp_z),
#         'roughness_vassilakis': unwrap(rough_v),
#         'roughness_sethares': unwrap(rough_s),
#         'weighted_spectral_centroid': unwrap(Loudness_SC_Hz),
#         'spectral_centroid_bark2hz': unwrap(centroid_bark2hz),
#         'time_MPS': unwrap(time_MPS),
#         'time_Zwicker': unwrap(time_Zwicker),
#         'time_vassilakis': unwrap(time_v),
#         'time_sethares': unwrap(time_s),
#         'time_bark2hz': unwrap(time_bark2hz),
#     }

#     # Interpolate features to target timestamps
#     interpolated = {
#         'MPS_roughness': interpolate_series(target_timestamps, features['time_MPS'], features['MPS_roughness']),
#         'roughness_Zwicker': interpolate_series(target_timestamps, features['time_Zwicker'], features['roughness_Zwicker']),
#         'sharpness_Zwicker': interpolate_series(target_timestamps, features['time_Zwicker'], features['sharpness_Zwicker']),
#         'roughness_vassilakis': interpolate_series(target_timestamps, features['time_vassilakis'], features['roughness_vassilakis']),
#         'roughness_sethares': interpolate_series(target_timestamps, features['time_sethares'], features['roughness_sethares']),
#         'weighted_spectral_centroid': interpolate_series(target_timestamps, features['time_Zwicker'], features['weighted_spectral_centroid']),
#         'spectral_centroid_bark2hz': interpolate_series(target_timestamps, features['time_bark2hz'], features['spectral_centroid_bark2hz']),
#         'timestamps': target_timestamps
#     }

#     return interpolated
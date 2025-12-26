import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from scipy.signal import savgol_filter
from skimage.restoration import denoise_bilateral
from statsmodels.nonparametric.smoothers_lowess import lowess

# Use this function to filter features
def filter_features(features,filter_length):

    filtered_features = {}

    for key,value in features.items():
        if key == "timestamp":
            filtered_features[key] = value
        else:
            filtered_features[key] = medfilt(value,filter_length)

    return filtered_features

# Use this function to plot features
def plot_features(features,overlap,save_path = None):

    start = time.time()

    # Median Filter
    apply_median_filter = True; min_median_filter_length = 3; max_median_filter_length = 11

    # SG Filter
    apply_sg_filter = False; min_sg_filter_length = 5; max_sg_filter_length = 13; polyorder = 2

    # LOWESS Filter
    apply_lowess_filter = False; frac = 0.02; it = 2

    # Bilateral Filter
    apply_bilateral_filter = False; sigma_color = 0.2; sigma_spatial = 3

    # Hard Clipping (frontend operation)
    apply_hard_clipping = True; lower_percentile = 5; upper_percentile = 95
    
    # Soft Clipping (frontend operation)
    apply_soft_clipping = True; scale = 10

    print("Set breakpoint here to change parameters live in the debug console")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"plots")
    os.makedirs(save_path,exist_ok = True)

    exclude = ["loudness_periodicity","mir_mps_roughness","mir_roughness_vassilakis","mir_sharpness_zwicker","perceived_pitch_f0_or_SC_weighted","weighted_spectral_centroid","yin_periodicity"]
    timestamps = features["timestamp"]
    for key,value in features.items():

        if key == "timestamp" or key in exclude:
            continue
        data = value.copy()

        plt.figure(figsize = (16,8))
        plt.plot(timestamps,data,label = "raw")
        plt.hlines(np.mean(data),timestamps[0],timestamps[-1],color = 'b',label = 'mean')
        plt.hlines(np.median(data),timestamps[0],timestamps[-1],color = 'r',label = 'median')
        plt.hlines((np.min(data) + np.max(data))/2,timestamps[0],timestamps[-1],color = 'k',label = 'midrange')

        if apply_median_filter:
            window_length = calculate_optimal_length(overlap,min_median_filter_length,max_median_filter_length)
            data = medfilt(data,window_length)
            plt.plot(timestamps,data,linestyle = "--",label = f"median filter ({window_length})")

        # if apply_sg_filter:
        #     min_value = np.min(data)
        #     max_value = np.max(data)
        #     window_length = calculate_optimal_length(overlap,min_sg_filter_length,max_sg_filter_length)
        #     data = savgol_filter(data,window_length,polyorder)
        #     data = clip(data,min_value,max_value)
        #     plt.plot(timestamps,data,linestyle = "--",label = f"sg filter ({window_length})")

        # if apply_median_filter and apply_sg_filter:
        #     min_value = np.min(data_median)
        #     max_value = np.max(data_median)
        #     window_length = calculate_optimal_length(overlap,min_sg_filter_length,max_sg_filter_length)
        #     data = savgol_filter(data_median,window_length,polyorder)
        #     data = clip(data,min_value,max_value)
        #     plt.plot(timestamps,data,linestyle = "--",label = f"sg filter ({window_length})")

        # if apply_lowess_filter:
        #     data = lowess(data,timestamps,frac = frac,it = it,return_sorted = False)
        #     plt.plot(timestamps,data,linestyle = "--",label = f"lowess filter ({frac})")

        # if apply_bilateral_filter:
        #     for sigma_spatial in [1,2,3,4,5,6,7]:
        #         data = denoise_bilateral(data,sigma_color = sigma_color,sigma_spatial = sigma_spatial)
        #         plt.plot(timestamps,data,linestyle = "--",label = f"bilateral filter ({sigma_spatial})")

        if apply_hard_clipping:
            data = hard_clip(data,lower_percentile,upper_percentile)
            plt.plot(timestamps,data,linestyle = "--",label = "hard clip")

        if apply_soft_clipping:
            shift = interpolate(np.median(data),np.min(data),np.max(data),0,1)
            data = soft_clip(data,shift,scale)
            plt.plot(timestamps,data,linestyle = "--",label = "soft clip")

        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc = "upper left")
        plt.title(key)
        plt.savefig(os.path.join(save_path,key + ".png"))

    end = time.time()

    print("Elapsed:",end - start,"seconds")

def clip(data,lower_bound,upper_bound):

    clipped_data = np.maximum(lower_bound,np.minimum(data,upper_bound))

    return clipped_data

def interpolate(x,x1,x2,y1,y2):

    x = np.asarray(x)
    if x2 != x1:
        y = y1 + (x - x1)*((y2 - y1)/(x2 - x1))
    else:
        y = np.full_like(x,(y1 + y2)/2)
        
    return y

def calculate_optimal_length(overlap,min_length = 3,max_length = 11):

    length = interpolate(overlap,0,1,min_length,max_length)
    length = int(np.ceil(length))
    length = length if length % 2 else length + 1

    return length

# Hard Clipping
def hard_clip(data,lower_percentile = 5,upper_percentile = 95):

    length = len(data)
    sorted_data = np.sort(data)

    if lower_percentile == upper_percentile:
        index = int(np.round((lower_percentile/100)*(length - 1)))
        lower_index = index
        upper_index = index
    else:
        lower_index = int(np.floor((lower_percentile/100)*(length - 1)))
        upper_index = int(np.ceil((upper_percentile/100)*(length - 1)))

    lower_bound = sorted_data[lower_index]
    upper_bound = sorted_data[upper_index]

    clipped_data = clip(data,lower_bound,upper_bound)

    return clipped_data

# Soft Clipping
def soft_clip(data,shift = 0.5,scale = 10):

    min_value = np.min(data)
    max_value = np.max(data)

    if max_value != min_value:

        normalized_data = interpolate(data,min_value,max_value,0,1)

        if scale != 0:
            compressor = lambda x: np.tanh(scale*(x - shift))
            compressed_data = compressor(normalized_data)
            compressed_data = interpolate(compressed_data,compressor(0),compressor(1),0,1)
        else:
            compressed_data = normalized_data

        clipped_data = interpolate(compressed_data,0,1,min_value,max_value)
    
    else:
        clipped_data = np.full(len(data),(min_value + max_value)/2)

    return clipped_data
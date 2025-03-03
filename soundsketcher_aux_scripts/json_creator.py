import os
import pandas as pd
import statistics
from scipy.stats import iqr
import matplotlib.pyplot as plt

#TODO/DONE 1 Log spectral centroid instead of linear & reducing the frequency range heigh (8000) 
#TODO/DONE 2 Nothing in user choices
#TODO/DONE 3 Y Axis
#TODO/DONE 4 Pitch --> YIN F0
#TODO/DONE 5 Spectral Deviation

#TODO New feature spectral centroid - standard deviation/2 
#TODO 


def creator(dir):

    # Directory containing CSV files
    folder_path = "../" + dir

    # Function to read each CSV file into a separate pandas DataFrame
    def csv_to_dataframes(folder_path):
        dataframes = []
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        # Define sorting order based on keywords
        sorting_order = {
            "centroid": 0,
            "amplitude": 1,
            "zcr": 2,
            "kurtosis": 3,
            "flux": 4,
            "deviation": 5
        }
        
        # Custom sorting function
        def custom_sort(file_name):
            for keyword in sorting_order.keys():
                if keyword in file_name:
                    return sorting_order[keyword]
            # If none of the keywords found, return a high value to place it at the end
            return len(sorting_order)

        # Sort CSV files based on keywords
        csv_files.sort(key=custom_sort)
        # print(csv_files)
        for csv_file in csv_files:
            csv_file_path = os.path.join(folder_path, csv_file)
            dataframe = pd.read_csv(csv_file_path, header=None)
            dataframes.append(dataframe)

        return dataframes

    def normalize(data):
        # Find the minimum and maximum values in the data
        min_val = min(data)
        max_val = max(data)
        
        # Normalize each value in the data
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        
        return normalized_data

    def median(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
        else:
            return sorted_lst[mid]
    

    def merge_and_fill_values(df1, df2):
        # Merge the datasets using outer join to keep all timestamps
        merged_data = pd.merge(df1, df2, on='Time', how='outer')

        # Fill missing values in Value2 column with 0
        merged_data['Value2'] = merged_data['Value2'].fillna(0)

        # Return the values from the second DataFrame as a list
        return merged_data['Value2'].tolist()
    

    def scale_array(array, min_value, max_value):
        # Find the minimum and maximum values of the array
        array_min = min(array)
        array_max = max(array)
        
        # Calculate the scaling factor
        scaling_factor = (max_value - min_value) / (array_max - array_min)
        
        # Calculate the offset
        offset = min_value - array_min * scaling_factor
        
        # Scale the array to the desired range
        scaled_array = [value * scaling_factor + offset for value in array]
        
        return scaled_array

    # Call the function to read CSV files into dataframes
    dataframes_list = csv_to_dataframes(folder_path)

    # Assign column names
    
    # print(dataframes_list)


    # "centroid": 0,
    # "amplitude": 1,
    # "zcr": 2,
    # "kurtosis": 3,
    # "flux": 4,
    # "deviation": 5

    dataframes_list[1].columns = ['Time', 'Value1']
    dataframes_list[0].columns = ['Time', 'Value2']
    dataframes_list[3].columns = ['Time', 'Value2']
    #dataframes_list[6].columns = ['Time', 'Value2']
    # print("LEN",len(dataframes_list))
    dataframes_list[5].columns = ['Time', 'Value2']
    

    # Extract specific columns from each dataframe
    timestamp = dataframes_list[1].iloc[:, 0].tolist()
    amplitude = dataframes_list[1].iloc[:, 1].tolist()  # Assuming amplitude is in the second column of the first dataframe
    spectral_centroid = merge_and_fill_values(dataframes_list[1], dataframes_list[0])  # Assuming spectral centroid is in the second column of the second dataframe
    spectral_flux = dataframes_list[4].iloc[:, 1].tolist()
    spectral_kurtosis = merge_and_fill_values(dataframes_list[1], dataframes_list[3])
    zerocrossingrate = dataframes_list[2].iloc[:, 1].tolist() # Assuming zero crossing rate is in the second column of the third dataframe
    #yin_f0 = merge_and_fill_values(dataframes_list[1], dataframes_list[6])
    standard_deviation = merge_and_fill_values(dataframes_list[1], dataframes_list[5])

    

    # Sample data
   

    





    # Assuming feature.zerocrossingrateArray is a list containing all zero crossing rate values
    median_amplitude = median(amplitude)
    median_spectral_centroid = median(spectral_centroid)
    median_zcr = median(zerocrossingrate)
    median_spectral_kurtosis = median(spectral_kurtosis)
    median_spectral_flux = median(spectral_flux)


    
    # Calculate the z-score for each data point
    zerocrossingrate_norm = [(x - median_zcr) / (1.349 * iqr(zerocrossingrate)) for x in zerocrossingrate]
    # Scale the z-scores from 0 to 90
    zerocrossingrate_norm_scaled = scale_array(zerocrossingrate_norm, 0, 45) 
    # print("MAX: ",max(zerocrossingrate_norm_scaled))
    # print("MAX: ",max(zerocrossingrate_norm))

    # Plot the data
    plt.plot(zerocrossingrate_norm_scaled)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Chart')
    plt.savefig('zcr_chart.png')
    # Construct JSON data
    # Create a dictionary with the median values
    json_data = {
        "Song": {
            "features_per_timestamp": [],
            "general_info": {
                "median_amplitude": median_amplitude,
                "median_spectral_centroid": median_spectral_centroid,
                "median_spectral_kurtosis": median_spectral_kurtosis,
                "median_spectral_flux": median_spectral_flux,
                "median_zcr": median_zcr,
                "iqr_zcr": iqr(zerocrossingrate)

            }
        }
    }
    for tms, amp, centroid, zcr, flux, kurtosis,  standard_deviation in zip(timestamp, amplitude, spectral_centroid, zerocrossingrate_norm_scaled, spectral_flux, spectral_kurtosis,  standard_deviation):
        json_data["Song"]["features_per_timestamp"].append({
            "timestamp": tms,
            "amplitude": amp,
            "spectral_centroid": centroid,
            "spectral_kurtosis": kurtosis,
            "spectral_flux" : flux,
            "zerocrossingrate": zcr,
            # "yin_f0" : yin_f0,
            "standard_deviation": standard_deviation,
            "brightness": (centroid + flux + zcr) / 3

        })
    # print(json_data)
    return json_data

def creator_librosa(features):
    # Extract individual features from the in-memory dictionary
    timestamp = features['timestamp']
    amplitude = features['rms']
    spectral_centroid = features['spectral_centroid']
    spectral_flux = features['spectral_flux']
    zerocrossingrate = features['zcr']
    spectral_flatness = features['spectral_flatness']
    f0 = features['f0_librosa']  # Librosa's F0
    voiced_prob = features['voiced_prob']  # Optional: Add voiced probability
    spectral_bandwidth = features['spectral_bandwidth']
    aubio_f0 = features.get('aubio_f0', [])  # Handle Aubio F0 safely
    crepe_f0 = features.get('crepe_f0', [])  # Handle CREPE F0
    crepe_confidence = features.get('crepe_confidence', [])  # Handle CREPE confidence
    yin_periodicity = features['yin_periodicity']
    f0_candidates = features['f0_candidates']
    loudness = features['loudness']
    sharpness = features['sharpness']
    # roughness = features.get['roughness'∫]
    print(f0_candidates)
    # Helper functions
    def normalize(data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    def median(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        mid = n // 2
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2 if n % 2 == 0 else sorted_lst[mid]

    def scale_array(array, min_value, max_value):
        array_min = min(array)
        array_max = max(array)
        scaling_factor = (max_value - min_value) / (array_max - array_min)
        offset = min_value - array_min * scaling_factor
        return [value * scaling_factor + offset for value in array]

    # Median and IQR calculations
    median_amplitude = median(amplitude)
    median_spectral_centroid = median(spectral_centroid)
    median_zcr = median(zerocrossingrate)
    median_spectral_flux = median(spectral_flux)
    median_flatness = median(spectral_flatness)
    # median_f0_librosa = median(f0[f0 > 0]) if any(f0 > 0) else 0  # Exclude unvoiced (zeros)
    # median_aubio_f0 = median(aubio_f0) if aubio_f0 else 0  # Handle empty Aubio F0
    # median_crepe_f0 = median(crepe_f0) if crepe_f0 else 0  # Handle empty CREPE F0

    # Z-score normalization for ZCR
    zerocrossingrate_norm = [(x - median_zcr) / (1.349 * iqr(zerocrossingrate)) for x in zerocrossingrate]
    zerocrossingrate_norm_scaled = scale_array(zerocrossingrate_norm, 0, 45)

    # # Visualization (optional)
    # plt.plot(zerocrossingrate_norm_scaled)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Line Chart (ZCR)')
    # plt.savefig('zcr_chart.png')

    # Construct JSON data
    json_data = {
        "Song": {
            "features_per_timestamp": [],
            "general_info": {
                "median_amplitude": median_amplitude,
                "median_spectral_centroid": median_spectral_centroid,
                "median_zcr": median_zcr,
                "median_spectral_flux": median_spectral_flux,
                "median_flatness": median_flatness,
                # "median_f0_librosa": median_f0_librosa,
                # "median_f0_aubio": median_aubio_f0,
                # "median_f0_crepe": median_crepe_f0,
                "iqr_zcr": iqr(zerocrossingrate),
            }
        }
    }

    # Combine features per timestamp
    for tms, amp, centroid, zcr, flux, flatness, f0_val, voiced_prob_val, bandwidth, aubio_f0_val, crepe_f0_val, crepe_conf_val, periodicity, f0_cand, loud, sharp in zip(
        timestamp, amplitude, spectral_centroid, zerocrossingrate_norm_scaled,
        spectral_flux, spectral_flatness, f0, voiced_prob, spectral_bandwidth,
        aubio_f0, crepe_f0, crepe_confidence, yin_periodicity, f0_candidates, loudness, sharpness
    ):
        json_data["Song"]["features_per_timestamp"].append({
            "timestamp": tms,
            "amplitude": amp,
            "spectral_centroid": centroid,
            "zerocrossingrate": zcr,
            "spectral_flux": flux,
            "spectral_flatness": flatness,
            "yin_f0_librosa": f0_val,
            "voiced_prob": voiced_prob_val,
            "yin_f0_aubio": aubio_f0_val,
            "crepe_f0": crepe_f0_val,
            "crepe_confidence": crepe_conf_val,
            "spectral_bandwidth": bandwidth,
            "brightness": (centroid + flux + zcr) / 3,
            "yin_periodicity": periodicity,
            "f0_candidates": f0_cand,
            "loudness": loud,
            "sharpness": sharp
        })

    return json_data

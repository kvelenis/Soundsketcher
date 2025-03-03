import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.ndimage import median_filter
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
from dtw import accelerated_dtw
import ruptures as rpt
import seaborn as sns
import librosa
import librosa.display

from collections import Counter

def cluster_objectification(features, audio_file_path, window_size, hop_length):
    # Step 1: Convert JSON to DataFrame
    df = pd.DataFrame(features)

    # Step 2: Extract features for clustering
    feature_columns = [
        "amplitude", 
        "spectral_centroid", 
        "zerocrossingrate", 
        "spectral_flux",
        "spectral_flatness", 
        "crepe_f0", 
        "spectral_bandwidth",
        "loudness",
        "sharpness",
        "brightness"
    ]
    feature_matrix = df[feature_columns].values

    # Step 3: Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)

    # Step 4: Dimensionality Reduction (UMAP for non-linear relationships)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_features = reducer.fit_transform(normalized_features)
    print(f"Reduced features shape (UMAP): {reduced_features.shape}")

    # Step 5: Determine Optimal Clusters Using Silhouette Analysis
    silhouette_scores = []
    best_silhouette_score = -1
    optimal_clusters = 2
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(reduced_features)
        sil_score = silhouette_score(reduced_features, labels)
        silhouette_scores.append(sil_score)
        if sil_score > best_silhouette_score:
            best_silhouette_score = sil_score
            optimal_clusters = k

    print(f"Optimal Number of Clusters (Silhouette): {optimal_clusters}")
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_features)
    df["first_level_cluster"] = labels

    # Step 6: Temporal Smoothing for First-Level Clustering
    smoothed_labels = median_filter(labels, size=25)
    df["smoothed_first_level_cluster"] = smoothed_labels

    # Visualize clusters on audio waveform
    y, sr = librosa.load(audio_file_path, sr=None)
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.scatter(
        df["timestamp"], 
        df["smoothed_first_level_cluster"], 
        c=df["smoothed_first_level_cluster"], 
        cmap="viridis", 
        s=10
    )
    plt.title("First-Level Clusters Overlayed on Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("waveform_with_first_level_clusters.png")
    plt.close()

    # Step 7: Second-Level Clustering (Hierarchical or Agglomerative)
    for cluster_label in np.unique(smoothed_labels):
        cluster_data = df[df["smoothed_first_level_cluster"] == cluster_label]
        if cluster_data.empty:
            continue

        # Use subset of features for second-level clustering
        sub_feature_columns = ["spectral_centroid", "zerocrossingrate", "amplitude", "spectral_flux"]
        sub_feature_matrix = cluster_data[sub_feature_columns].values

        # Normalize and reduce dimensions for sub-clustering
        sub_normalized_features = scaler.fit_transform(sub_feature_matrix)
        sub_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
        sub_reduced_features = sub_reducer.fit_transform(sub_normalized_features)

        # Apply Hierarchical Clustering
        linkage_matrix = linkage(sub_reduced_features, method="ward")
        distance_threshold = 0.7 * np.max(linkage_matrix[:, 2])
        hierarchical_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
        cluster_data["second_level_cluster"] = hierarchical_labels

        # Temporal Smoothing for Second-Level Clustering
        smoothed_hierarchical_labels = median_filter(hierarchical_labels, size=25)
        cluster_data["smoothed_second_level_cluster"] = smoothed_hierarchical_labels

        # Map back to main DataFrame
        df.loc[cluster_data.index, "second_level_cluster"] = hierarchical_labels
        df.loc[cluster_data.index, "smoothed_second_level_cluster"] = smoothed_hierarchical_labels

    # Visualize second-level clusters on audio waveform
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.scatter(
        df["timestamp"], 
        df["smoothed_second_level_cluster"], 
        c=df["smoothed_second_level_cluster"], 
        cmap="plasma", 
        s=10
    )
    plt.title("Second-Level Clusters Overlayed on Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("waveform_with_second_level_clusters.png")
    plt.close()

    # Save results
    df.to_csv("clustered_features_with_second_level_clusters.csv", index=False)
    print("Clustered features with second-level clusters saved.")

    return df

def cluster_objectification_with_dtw(features, audio_file_path):
    # Step 1: Convert JSON to DataFrame
    df = pd.DataFrame(features)

    # Define feature sets for first and second-level clustering
    first_level_features = ["amplitude", "spectral_centroid", "zerocrossingrate", "brightness"]
    second_level_features = ["sharpness", "loudness", "roughness", "spectral_flux"]

    # First-Level Clustering
    first_level_matrix = df[first_level_features].values
    first_scaler = StandardScaler()
    normalized_first_features = first_scaler.fit_transform(first_level_matrix)

    # Dimensionality Reduction (UMAP)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_first_features = reducer.fit_transform(normalized_first_features)

    # Silhouette Analysis for First-Level Clustering
    silhouette_scores = []
    optimal_clusters = 2
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(reduced_first_features)
        sil_score = silhouette_score(reduced_first_features, labels)
        silhouette_scores.append(sil_score)
        if sil_score == max(silhouette_scores):
            optimal_clusters = k

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_first_features)
    df["first_level_cluster"] = labels

    # Temporal Smoothing
    smoothed_labels = median_filter(labels, size=5)
    df["smoothed_first_level_cluster"] = smoothed_labels

    # DTW Analysis within Each First-Level Cluster
    for cluster_label in np.unique(smoothed_labels):
        cluster_data = df[df["smoothed_first_level_cluster"] == cluster_label]
        if cluster_data.empty:
            continue

        # Calculate DTW Distance Matrix
        feature_columns = first_level_features  # Use first-level features for DTW
        dtw_matrix = np.zeros((len(feature_columns), len(feature_columns)))
        for i, feature1 in enumerate(feature_columns):
            for j, feature2 in enumerate(feature_columns):
                dist, _, _, _ = accelerated_dtw(cluster_data[feature1].values.reshape(-1, 1),
                                                cluster_data[feature2].values.reshape(-1, 1),
                                                dist='euclidean')
                dtw_matrix[i, j] = dist

        # Visualize DTW Distance Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(dtw_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=feature_columns, yticklabels=feature_columns)
        plt.title(f"DTW Similarity Matrix for First-Level Cluster {cluster_label}")
        plt.savefig(f"dtw_similarity_cluster_{cluster_label}.png")
        plt.close()

        # Use DTW for Nested Clustering
        hierarchical_labels = fcluster(linkage(dtw_matrix, method='ward'), t=1.5, criterion='distance')
        df.loc[cluster_data.index, "nested_cluster"] = hierarchical_labels

        # Temporal Smoothing for Nested Clusters
        smoothed_nested_labels = median_filter(hierarchical_labels, size=5)
        df.loc[cluster_data.index, "smoothed_nested_cluster"] = smoothed_nested_labels

    # Save Results
    df.to_csv("clustered_features_with_dtw.csv", index=False)
    print("Clustered features with DTW saved to 'clustered_features_with_dtw.csv'")

    return df
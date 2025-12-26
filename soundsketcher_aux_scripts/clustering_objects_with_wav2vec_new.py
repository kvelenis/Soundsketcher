#
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from cuml.cluster import KMeans as cuKMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import umap.umap_ as umap
import librosa
import librosa.display
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from torch import nn
from scipy.ndimage import median_filter
import ruptures as rpt

import cudf
import cupy as cp

from sklearn.metrics.pairwise import cosine_similarity
from transformers import ClapModel, ClapProcessor

import laion_clap


def objectifier(audio_path):
        
    # Flags to control the dimensionality reduction techniques
    USE_UMAP = False  # Set to True to use UMAP
    USE_PCA = False    # Set to True to use PCA
    # Add a flag to enable or disable attention mechanism
    USE_ATTENTION = False
    DEBUG_PRINTS = True

    # Step 1: Add Attention Mechanism
    class AttentionLayer(nn.Module):
        """
        Self-Attention layer to dynamically weight time frames.
        """
        def __init__(self, embedding_dim, n_heads=4):
            super(AttentionLayer, self).__init__()
            self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads)

        def forward(self, embeddings):
            embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
            attended_embeddings, _ = self.attention(embeddings, embeddings, embeddings)
            return attended_embeddings.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)

    # Step 2: Extract Wav2Vec Embeddings
    def extract_wav2vec_embeddings(audio_path, model_name="facebook/wav2vec2-base"):
        if DEBUG_PRINTS:
            print("Step 2: Extracting Wav2Vec Embeddings")
        """
        Extract Wav2Vec2 embeddings from raw audio.
        """
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)

        y, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            embeddings = model(input_values).last_hidden_state  # (batch_size, seq_len, embedding_dim)
        
        return embeddings.squeeze(0).numpy(), sr, y  # (seq_len, embedding_dim)



    # Step 3: Apply Attention to Embeddings
    def apply_attention_to_embeddings(embeddings):
        if DEBUG_PRINTS:
            print("Step 3: Applying Attention to Embeddings")
        """
        Apply a self-attention layer to refine embeddings if USE_ATTENTION is True.
        """
        if USE_ATTENTION:
            print("Using attention mechanism...")
            embedding_dim = embeddings.shape[1]
            attention_layer = AttentionLayer(embedding_dim=embedding_dim)
            embeddings_tensor = torch.tensor(embeddings).unsqueeze(0)  # Add batch dimension
            attended_embeddings = attention_layer(embeddings_tensor).squeeze(0).detach().numpy()
            return attended_embeddings
        else:
            print("Skipping attention mechanism, using original embeddings...")
            return embeddings  # Return embeddings as-is

    # Step 4: Reduce Dimensionality
    def reduce_embeddings(embeddings, n_components=2):
        if DEBUG_PRINTS:
            print("Step 4: Reducing Dimensionality simple")

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    # Step 4: Reduce Dimensionality with UMAP or PCA
    def reduce_embeddings(embeddings, n_components=2):
        if DEBUG_PRINTS:
            print("Step 4: Reducing Dimensionality")
        """
        Reduce dimensionality of embeddings using UMAP or PCA.
        """
        if USE_UMAP:
            print("Using UMAP for dimensionality reduction...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        elif USE_PCA:
            print("Using PCA for dimensionality reduction...")
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            print("Skipping dimensionality reduction, using original embeddings...")
            reduced_embeddings = embeddings  # No dimensionality reduction applied

        return reduced_embeddings

    # Step 5: Dynamic Cluster Determination with Silhouette or Gap Statistic
    def determine_optimal_clusters(embeddings, max_clusters=20, method="silhouette"):
        if DEBUG_PRINTS:
            print("Step 5: Dynamic Clustering Determination with, ", method)
        """
        Determine the optimal number of clusters using Silhouette Analysis or Gap Statistic.
        
        Parameters:
        - embeddings: array-like, shape (n_samples, n_features)
            Data to be clustered.
        - max_clusters: int, default=20
            Maximum number of clusters to evaluate.
        - method: str, default="silhouette"
            Method to use for cluster determination. Options: "silhouette", "gap".
        
        Returns:
        - optimal_clusters: int
            Optimal number of clusters.
        """
        if method == "silhouette":
            silhouette_scores = []
            best_silhouette_score = -1
            optimal_clusters = 2

            for k in range(2, max_clusters):
                kmeans = cuKMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                sil_score = silhouette_score(embeddings, labels)
                silhouette_scores.append(sil_score)

                if sil_score > best_silhouette_score:
                    best_silhouette_score = sil_score
                    optimal_clusters = k

            print(f"Optimal Number of Clusters (Silhouette): {optimal_clusters}")
            return optimal_clusters

        elif method == "gap":
            # Gap Statistic Calculation
            def compute_gap_statistic(data, k, n_reference=10):
                # Convert original data to cuDF if it's not already
                if not isinstance(data, cudf.DataFrame):
                    data = cudf.DataFrame(data)

                # Fit KMeans on the actual data
                kmeans = cuKMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                actual_inertia = kmeans.inertia_

                # Generate random reference datasets and compute their inertias
                reference_inertias = []
                for i in range(n_reference):
                    reference_data = shuffle(data.to_pandas(), random_state=42 + i)  # shuffle is CPU-side
                    reference_data = cudf.DataFrame(reference_data)

                    kmeans_ref = cuKMeans(n_clusters=k, random_state=42 + i)
                    kmeans_ref.fit(reference_data)
                    reference_inertias.append(kmeans_ref.inertia_)

                # Compute the gap statistic
                gap = np.mean(np.log(reference_inertias)) - np.log(actual_inertia)
                return gap

            gap_statistics = []
            optimal_clusters = 2

            for k in range(2, max_clusters):
                gap = compute_gap_statistic(embeddings, k)
                gap_statistics.append(gap)

            # Find the optimal k using the largest gap
            optimal_clusters = np.argmax(gap_statistics) + 2
            print(f"Optimal Number of Clusters (Gap Statistic, GPU): {optimal_clusters}")
            return optimal_clusters

        else:
            raise ValueError("Invalid method. Choose 'silhouette' or 'gap'.")

    # Step 6: Perform Clustering
    def cluster_embeddings(embeddings, n_clusters):
        if DEBUG_PRINTS:
            print("Step 6: Performing Clustering")
        kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels

    # Step 7: Add Temporal Smoothing
    def smooth_labels(labels, size=20):
        if DEBUG_PRINTS:
            print("Step 7: Adding Clustering")
        smoothed_labels = median_filter(labels, size=size)
        return smoothed_labels

    # Step 8: Detect Change Points
    def detect_change_points(embeddings):
        if DEBUG_PRINTS:
            print("Step 8: Detecting Change Points")

        model = rpt.Pelt(model="rbf").fit(embeddings)
        breakpoints = model.predict(pen=10)
        return breakpoints

    # Step 9: Visualize Clusters and Change Points with Non-Overlapping Similarity Annotations and Legend
    def visualize_clusters_and_transitions(audio_path, sr, y, labels, breakpoints, hop_length, similarities):
        if DEBUG_PRINTS:
            print("Step 9: Visualizing Clusters")
        """
        Visualize clusters, change points, and annotate similarities between segments with non-overlapping text,
        including a legend for cluster colors.
        """
        times = librosa.frames_to_time(range(len(labels)), sr=sr, hop_length=hop_length)

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(y, sr=sr)

        # Create a color map for clusters
        unique_labels = np.unique(labels)
        cluster_colors = {label: f"C{label % 10}" for label in unique_labels}

        # Plot cluster transitions
        for t, label in zip(times, labels):
            plt.axvline(x=t, color=cluster_colors[label], linestyle='--', alpha=0.5)

        # Plot change points and annotate similarities
        for i, bp in enumerate(breakpoints[:-1]):  # Exclude last breakpoint since it's the end of the audio
            bp_time = bp * hop_length / sr
            plt.axvline(x=bp_time, color="red", linestyle="-", label="Change Point" if i == 0 else "")
            
            # Alternate text height to avoid overlap
            height_factor = 0.9 if i % 2 == 0 else 0.8  # Alternate height between 90% and 80% of the waveform's max amplitude
            similarity_text = f"{similarities[i]:.2f}"  # Format similarity to 2 decimal places
            plt.text(bp_time, max(y) * height_factor, similarity_text, color="black", fontsize=10, ha="center")

        # Add a legend for cluster colors
        for label, color in cluster_colors.items():
            plt.axvline(x=-1, color=color, linestyle='--', label=f"Cluster {label}")  # Dummy line for legend

        plt.title("Clusters and Change Points Overlayed on Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        
        # plt.savefig("/mnt/ssd1/kvelenis/soundsketcher/clustering_destination_plots/"+os.path.splitext(audio_path)[0]+"_"+"clusters_and_transitions_with_non_overlapping_similarities_and_legend.png")
        # plt.show()


    # Step 11: Extract CLAP Embeddings for Segments
    def extract_clap_embeddings_for_segments(audio_path, breakpoints, sr, hop_length, processor):
        if DEBUG_PRINTS:
            print("Step 11: Extracting CLAP Embeddings for Segments")
        """
        Extract CLAP embeddings for each segmented part of the audio.
        """
        # Load CLAP model and processor
        # Initialize CLAP model and processor
        model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        print(model.config)
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Resample to 48kHz for CLAP model compatibility
        target_sr = 48000
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Create audio segments based on breakpoints
        segments = [y_resampled[int(bp * hop_length * (target_sr / sr)):int(next_bp * hop_length * (target_sr / sr))]
                    for bp, next_bp in zip([0] + breakpoints[:-1], breakpoints)]

        # Extract embeddings for each segment
        embeddings = []
        for segment in segments:
            # Ensure the segment is passed in the correct format
            inputs = processor(audios=segment, sampling_rate=target_sr, return_tensors="pt")
            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs)  # Extract CLAP audio embeddings
            embeddings.append(audio_embed.detach().cpu().numpy())  # Detach and convert to numpy

        return embeddings




        # # Extract embeddings for each segment
        # embeddings = []
        # for segment in segments:
        #     inputs = processor(audio=segment, return_tensors="pt", sampling_rate=sr)
        #     with torch.no_grad():
        #         embeddings.append(model.get_audio_features(**inputs).squeeze(0).numpy())

        # return embeddings

    # Step 12: Compare Consecutive Segments
    def compare_consecutive_segments_with_clap(clap_embeddings):
        if DEBUG_PRINTS:
            print("Step 12: Comparing Consecutive Segments")
        """
        Compare consecutive audio segments using CLAP embeddings.
        """
        # Reduce dimensionality of embeddings (e.g., average over sequence dimension)
        reduced_embeddings = [np.mean(embed, axis=0) for embed in clap_embeddings]

        # Calculate cosine similarities between consecutive embeddings
        similarities = [
            cosine_similarity([reduced_embeddings[i]], [reduced_embeddings[i + 1]])[0][0]
            for i in range(len(reduced_embeddings) - 1)
        ]
        return similarities

    # Step 13: Visualize CLAP Segment Similarities
    def visualize_clap_segment_similarity(similarities, breakpoints, sr, hop_length, audio_file_path):
        if DEBUG_PRINTS:
            print("Step 13: Visualizing CLAP Segment Similarities")
        """
        Visualize cosine similarities between consecutive audio segments.
        """
        times = librosa.frames_to_time(breakpoints, sr=sr, hop_length=hop_length)
        plt.figure(figsize=(15, 5))
        plt.plot(times[:-1], similarities, marker="o", label="Cosine Similarity")
        plt.title("Consecutive Segment Similarities (CLAP)")
        plt.xlabel("Time (s)")
        plt.ylabel("Similarity")
        plt.legend()
        plt.grid()
        # plt.savefig("clustering_destination_plots/"+os.path.splitext(audio_file_path)[0]+"_"+"segment_similarities.png")
        # print(similarities)
        # plt.show()

    # Step 15: Generate CLAP Text Embeddings for Semantic Labeling
    def generate_text_embeddings(terms, processor, model):
        if DEBUG_PRINTS:
            print("Step 15: Generating CLAP Text Embeddings for Semantic Labeling")
        """
        Generate CLAP embeddings for a list of textual terms.
        """
        text_inputs = processor(text=terms, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs)
        return terms, text_embeddings.numpy()

    # Step 15: Extract CLAP Embeddings for Clusters
    def extract_clap_embeddings_for_clusters(audio_path, labels, sr, hop_length, cluster_count, processor, model):
        if DEBUG_PRINTS:
            print("Step 15: Extracting CLAP Embeddings for Clusters")
        """
        Extract CLAP embeddings for audio segments belonging to each cluster.
        """
        

        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Resample to 48kHz for CLAP model compatibility
        target_sr = 48000
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Initialize cluster segments
        cluster_segments = {cluster: [] for cluster in range(cluster_count)}

        # Assign audio frames to clusters
        for i, label in enumerate(labels):
            start_sample = int(i * hop_length * (target_sr / sr))
            end_sample = int((i + 1) * hop_length * (target_sr / sr))
            cluster_segments[label].append(y_resampled[start_sample:end_sample])

        # Calculate embeddings for each cluster
        cluster_embeddings = {}
        for cluster, segments in cluster_segments.items():
            if len(segments) == 0:
                continue
            # Concatenate all segments of the cluster
            cluster_audio = np.concatenate(segments)
            inputs = processor(audios=cluster_audio, sampling_rate=target_sr, return_tensors="pt")
            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs)  # Extract CLAP audio embeddings
            cluster_embeddings[cluster] = audio_embed.detach().cpu().numpy()  # Detach and convert to numpy

        return cluster_embeddings

    # Step 18: Compute Centroids of CLAP Embeddings
    def compute_cluster_centroids(cluster_embeddings):
        if DEBUG_PRINTS:
            print("Step 18: Computing Centroids of CLAP Embeddings")
        """
        Compute the centroid of CLAP embeddings for each cluster.
        """
        cluster_centroids = {}
        for cluster, embeddings in cluster_embeddings.items():
            # Compute the mean embedding for the cluster
            cluster_centroids[cluster] = np.mean(embeddings, axis=0)
        return cluster_centroids

    # Step 19: Compare Cluster Centroids with Text Embeddings and Include Top Terms
    def compare_centroids_with_text(cluster_centroids, text_embeddings, terms, top_n=5):
        if DEBUG_PRINTS:
            print("Step 19: Compare Cluster Centroids with Text Embeddings and Include Top Terms")
        """
        Compare centroids of cluster CLAP embeddings with text embeddings.

        Parameters:
        - cluster_centroids: dict, cluster indices as keys and centroid embeddings as values.
        - text_embeddings: array-like, text embeddings corresponding to the terms.
        - terms: list of str, textual terms to compare.
        - top_n: int, number of top terms to return for each cluster.

        Returns:
        - semantic_centroids: dict, cluster indices as keys with information on terms and similarities.
        """
        semantic_centroids = {}
        for cluster, centroid in cluster_centroids.items():
            # Compute cosine similarities with text embeddings
            similarities = cosine_similarity([centroid], text_embeddings)[0]

            # Get top N similar terms
            top_indices = np.argsort(similarities)[::-1][:top_n]  # Indices of top N terms
            top_terms = [(terms[i], similarities[i]) for i in top_indices]

            # Store top terms and their similarities
            semantic_centroids[cluster] = {
                "embedding": centroid,
                "top_terms": top_terms  # List of tuples (term, similarity)
            }

        return semantic_centroids

    def display_top_terms_for_clusters(semantic_centroids, terms, text_embeddings, audio_file_path=None):
        """
        Display the top 5 terms with the highest similarity for each cluster and save to a text file.

        Parameters:
        - semantic_centroids: dict, cluster indices as keys and centroid embeddings as values.
        - terms: list of str, the textual terms for comparison.
        - text_embeddings: array-like, text embeddings corresponding to the terms.
        - audio_file_path: str, path to the audio file (used for saving the output).
        """
        output_file_path = None
        if audio_file_path:
            output_file_path = "clustering_destination_plots/" + os.path.splitext(audio_file_path)[0] + "_top_terms.txt"

        if output_file_path:
            file = open(output_file_path, "w")

        for cluster_idx, centroid in semantic_centroids.items():
            # Calculate similarities with all text embeddings
            similarities = cosine_similarity([centroid], text_embeddings)[0]
            
            # Sort terms by similarity
            top_indices = np.argsort(similarities)[::-1][:5]  # Indices of top 5 terms
            top_terms = [(terms[i], similarities[i]) for i in top_indices]

            # Print cluster information
            cluster_info = f"Cluster {cluster_idx}:\n"
            print(cluster_info)
            if output_file_path:
                file.write(cluster_info)

            for rank, (term, similarity) in enumerate(top_terms, start=1):
                term_info = f"  {rank}. {term} (Similarity: {similarity:.2f})\n"
                print(term_info)
                if output_file_path:
                    file.write(term_info)

            print()

        if output_file_path:
            file.close()
            print(f"Top terms saved to {output_file_path}")

    def filter_small_clusters_and_reassign(embeddings, labels, min_size=40):
        """
        Reassign points from small clusters to the nearest larger cluster.

        Parameters:
        - embeddings: array-like, shape (n_samples, n_features)
            Embeddings of the data points.
        - labels: array-like, cluster labels for each point.
        - min_size: int, minimum size for a cluster to be considered valid.

        Returns:
        - filtered_labels: array-like, updated cluster labels.
        """
        from collections import Counter
        from sklearn.metrics.pairwise import euclidean_distances
        import numpy as np

        # Count the size of each cluster
        cluster_counts = Counter(labels)

        # Print cluster sizes
        print("Cluster sizes:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} items")

        # Find valid clusters (with size >= min_size)
        valid_clusters = {cluster for cluster, count in cluster_counts.items() if count >= min_size}
        print(f"Valid clusters (>= {min_size} items): {valid_clusters}")

        # Compute centroids for valid clusters
        valid_centroids = {
            cluster: embeddings[labels == cluster].mean(axis=0)
            for cluster in valid_clusters
        }

        # Reassign small cluster points
        filtered_labels = np.array(labels)  # Copy original labels
        for i, label in enumerate(labels):
            if label not in valid_clusters:  # If the point is in a small cluster
                # Compute distances to valid centroids
                distances = {
                    valid_label: euclidean_distances([embeddings[i]], [centroid])[0][0]
                    for valid_label, centroid in valid_centroids.items()
                }
                # Find the nearest valid cluster
                nearest_cluster = min(distances, key=distances.get)
                filtered_labels[i] = nearest_cluster  # Reassign to nearest cluster

        return filtered_labels

    def gather_plot_data_for_plotly(audio_path, sr, y, labels, cluster_regions, similarities, semantic_centroids, top_n=5):
        """
        Gather all plot data for visualization with Plotly and save it as a JSON file, excluding change points.

        Parameters:
        - audio_path: str, the path to the audio file.
        - sr: int, the sample rate of the audio.
        - y: np.ndarray, the waveform of the audio.
        - labels: array-like, cluster labels for each point.
        - cluster_regions: list of dicts, each containing start_time, end_time, and label.
        - similarities: list, cosine similarity values between segments.
        - semantic_centroids: dict, top terms for each cluster.
        - top_n: int, number of top terms to include for each cluster.
        """
        # Generate colors for each cluster in HEX
        unique_labels = np.unique(labels)
        cluster_colors = {label: f"#{np.random.randint(0, 0xFFFFFF):06x}" for label in unique_labels}

        # Prepare data for Plotly
        plot_data = {
            "audio_file": os.path.basename(audio_path),
            "sample_rate": sr,
            "waveform": y.tolist(),  # Convert waveform to a list for JSON compatibility
            "clusters": [],
            "similarities": []
        }

        # Add cluster information
        for label in unique_labels:
            # Get the top terms for this cluster
            top_terms = []
            if label in semantic_centroids:
                terms = semantic_centroids[label]["top_terms"]
                top_terms = [{"term": term, "similarity": float(sim)} for term, sim in terms[:top_n]]

            # Add cluster regions
            regions = [
                {
                    "start_time": region["start_time"],
                    "end_time": region["end_time"]
                }
                for region in cluster_regions if region["label"] == label
            ]

            # Add cluster data
            plot_data["clusters"].append({
                "label": int(label),
                "color": cluster_colors[label],
                "top_terms": top_terms,
                "regions": regions  # Add regions for this cluster
            })

        # # Save data to a JSON file
        # json_file_path = "clustering_destination_plots/" + os.path.splitext(audio_path)[0] + "_plotly_data.json"
        # with open(json_file_path, "w") as f:
        #     json.dump(plot_data, f, indent=4)

        # print(f"Plotly-friendly data without change points saved to {json_file_path}")
        return plot_data

    def detect_cluster_regions(labels, sample_rate, hop_length):
        """
        Detect continuous regions for clusters based on labels.
        No merging or minimum duration logic.

        Parameters:
        - labels: list of int, cluster labels for each frame.
        - sample_rate: int, audio sample rate.
        - hop_length: int, number of samples per frame.

        Returns:
        - cluster_regions: list of dict, each with start_time, end_time, and label.
        """
        cluster_regions = []
        current_label = labels[0]
        start_idx = 0

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # Create a region for the current label
                end_idx = i
                cluster_regions.append({
                    "start_time": start_idx * hop_length / sample_rate,
                    "end_time": end_idx * hop_length / sample_rate,
                    "label": current_label,
                })
                # Update for the new label
                current_label = labels[i]
                start_idx = i

        # Add the final region
        cluster_regions.append({
            "start_time": start_idx * hop_length / sample_rate,
            "end_time": len(labels) * hop_length / sample_rate,
            "label": current_label,
        })

        return cluster_regions


    def load_Clap_model(checkpoint_path):
        # Load CLAP model and processor
        model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        # Load the checkpoint and extract the state_dict
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        # Handle the checkpoint structure
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):  # If it's a flat dictionary without "state_dict"
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint does not contain 'state_dict'. Verify the checkpoint structure.")

        # Strip potential prefixes from state_dict keys (if saved with a prefix like 'module.')
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load the state_dict into the model
        model.load_state_dict(new_state_dict, strict=False)

        # Set the model to evaluation mode
        model.eval()
        print(f"Custom CLAP model loaded from: {checkpoint_path}")
        return processor, model

    def merge_similar_clusters(cluster_regions, cluster_clap_embeddings, similarity_threshold=0.6):
        """
        Merge consecutive clusters based on their CLAP embedding similarity.
        
        Parameters:
        - cluster_regions: list of dict, each with 'start_time', 'end_time', and 'label'.
        - cluster_clap_embeddings: dict of cluster embeddings.
        - similarity_threshold: float, threshold above which clusters will be merged.
        
        Returns:
        - merged_regions: list of dict, updated cluster regions.
        """
        merged_regions = []
        i = 0
        
        while i < len(cluster_regions):
            current_region = cluster_regions[i]
            current_label = current_region["label"]
            current_embedding = np.mean(cluster_clap_embeddings[current_label], axis=0)  # Average Pooling

            j = i + 1
            while j < len(cluster_regions):
                next_region = cluster_regions[j]
                next_label = next_region["label"]
                next_embedding = np.mean(cluster_clap_embeddings[next_label], axis=0)  # Average Pooling
                
                # Compute similarity
                similarity = cosine_similarity([current_embedding], [next_embedding])[0][0]

                if similarity >= similarity_threshold:
                    # Merge regions
                    current_region["end_time"] = next_region["end_time"]
                else:
                    break
                j += 1

            merged_regions.append(current_region)
            i = j

        return merged_regions


    # Step 14: Full Pipeline with CLAP Comparison
    def process_audio_with_clap_comparison(audio_file_path, hop_length=320, max_clusters=10, model_name="laion/clap"):
        if DEBUG_PRINTS:
            print("Step 14: Full Pipeline with CLAP Comparison")
        """
        Full pipeline with CLAP embedding comparison.
        """

        clap_processor, clap_model = load_Clap_model("/mnt/ssd1/kvelenis/soundsketcher/aux_models/music_speech_audioset_epoch_15_esc_89.98.pt")
        # Extract Wav2Vec embeddings
        embeddings, sr, y = extract_wav2vec_embeddings(audio_file_path)

        # Apply attention to embeddings
        embeddings = apply_attention_to_embeddings(embeddings)

        # Reduce dimensionality
        reduced_embeddings = reduce_embeddings(embeddings)

        # Determine optimal clusters
        # Method to use for cluster determination. Options: "silhouette", "gap".
        optimal_clusters = determine_optimal_clusters(embeddings, max_clusters=20, method="gap")

        # Perform clustering
        labels = cluster_embeddings(reduced_embeddings, n_clusters=optimal_clusters)
        # Reassign small clusters to nearest valid cluster
        filtered_labels = filter_small_clusters_and_reassign(reduced_embeddings, labels, min_size=30)

        # Apply temporal smoothing to stabilize labels over time
        smoothed_labels = smooth_labels(filtered_labels)
        # Apply temporal smoothing
        # smoothed_labels = smooth_labels(labels)
        # Detect cluster change points based on transitions
        cluster_regions = detect_cluster_regions(smoothed_labels, sample_rate=16000, hop_length=320)

        # Verify the results
        # for region in cluster_regions:
        #     print(region)
        # Detect change points
        breakpoints = detect_change_points(reduced_embeddings)

        # Extract CLAP embeddings for segments
        clap_embeddings = extract_clap_embeddings_for_segments(audio_file_path, breakpoints, sr, hop_length, clap_processor)
        # print(clap_embeddings)
        # Compare consecutive segments
        similarities = compare_consecutive_segments_with_clap(clap_embeddings)

        # Visualize similarities
        visualize_clap_segment_similarity(similarities, breakpoints, sr, hop_length, audio_file_path)

        # Visualize clusters and transitions
        visualize_clusters_and_transitions(audio_file_path, sr, y, smoothed_labels, breakpoints, hop_length ,similarities)


        # Generate CLAP text embeddings for semantic analysis
        text_terms = [
            "A bright sound is sharp, clear, and high in frequency. It stands out and feels crisp, often described as radiant or shimmering. Examples include cymbals, violins, or high piano notes.",
            "A dark sound is deep, muted, and rich in low frequencies. It often feels heavy, subdued, and moody. Examples include bass guitar, low brass, or ambient drones.",
            "A warm sound is rich, full, and pleasing to the ear. It has a balanced tonal quality with smooth mid and low frequencies. Examples include acoustic guitars, vocal harmonies, or a soft saxophone.",
            "A cold sound feels distant, sharp, and unemotional. It is often associated with thin or metallic tones and lacks warmth. Examples include synthetic pads, icy wind sounds, or high electronic tones.",
            "A rough sound is coarse, jagged, and textured. It feels unrefined or gritty, often with harsh edges. Examples include distorted guitars, gravel underfoot, or industrial machinery.",
            "A smooth sound is fluid, continuous, and free of abrupt changes. It feels polished and soothing. Examples include a cello melody, flowing water, or a soft breeze.",
            "A metallic sound has a resonant, ringing quality similar to struck metal. It often feels sharp and vibrant. Examples include bells, cymbals, or metal pipes.",
            "A soft sound is gentle, quiet, and unobtrusive. It often feels delicate and calming. Examples include a whisper, light footsteps, or rustling leaves.",
            "A granular sound has a fragmented or textured quality, often created by many small particles or grains. Examples include the crackle of a fire, the crunch of gravel, or digital granular synthesis.",
            "A high-pitched sound is characterized by high frequencies. It often feels sharp, thin, or piercing. Examples include a whistle, bird chirps, or a violin's upper register.",
            "A low-pitched sound is characterized by low frequencies. It feels deep, resonant, and powerful. Examples include bass notes, thunder, or a deep male voice.",
            "A harmonic sound is rich and pleasing, characterized by harmonious frequencies. Examples include a choir, an organ, or a well-tuned guitar chord.",
            "A disharmonic sound is dissonant, jarring, or unpleasant, often with clashing frequencies. Examples include off-key instruments, metal scraping, or chaotic industrial noise.",
            "A melodic sound is tuneful, flowing, and pleasing to the ear. Examples include a piano solo, a violin melody, or a bird song.",
            "A dissonant sound is harsh, clashing, and unresolved. It creates tension or unease. Examples include a horror movie score or an out-of-tune orchestra.",
            "A rhythmic sound has a structured, repetitive pattern that creates a beat or tempo. Examples include a drumbeat, hand claps, or a ticking clock.",
            "A chaotic sound is disordered, unpredictable, and lacking clear structure. Examples include a crowded marketplace, a thunderstorm, or a glitching electronic signal.",
            "A natural sound comes from the environment, often soothing and unprocessed. Examples include birdsong, rustling leaves, or ocean waves.",
            "A mechanical sound is artificial, repetitive, and associated with machines. Examples include gears turning, a ticking clock, or an engine.",
            "An urban sound captures the ambiance of a city, often a mix of various noises. Examples include traffic, footsteps on pavement, or distant sirens.",
            "A rainy sound evokes the atmosphere of rainfall, often calming and rhythmic. Examples include drops hitting a surface, gentle storms, or flowing water.",
            "A noise sound is unstructured, often random, and can range from background hums to static. Examples include white noise, crowd chatter, or static from a radio.",
            "A tonal sound has a clear pitch or tonal center. Examples include a musical note, a tuning fork, or a singing voice."
        ]
        # clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        # clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        terms, text_embeddings = generate_text_embeddings(text_terms, clap_processor, clap_model)


        # Extract CLAP embeddings for clusters
        cluster_clap_embeddings = extract_clap_embeddings_for_clusters(audio_file_path, smoothed_labels, sr, hop_length, optimal_clusters, clap_processor, clap_model)


        # Compute centroids of CLAP embeddings for clusters
        cluster_centroids = compute_cluster_centroids(cluster_clap_embeddings)
        # Merge similar clusters based on CLAP embeddings
        merged_regions = merge_similar_clusters(cluster_regions, cluster_clap_embeddings, similarity_threshold=0.6)
        # cluster_centroids = merged_regions
        # Compare cluster centroids with text embeddings
        semantic_centroids = compare_centroids_with_text(cluster_centroids, text_embeddings, text_terms, top_n=5)

        # Display top 5 semantic terms for each cluster
        # top_terms = display_top_terms_for_clusters(cluster_centroids, terms, text_embeddings, audio_file_path)
        # json_data = gather_plot_data_for_plotly(audio_path, sr, y, labels, breakpoints, hop_length, similarities, semantic_centroids, top_n=5)
        json_data = gather_plot_data_for_plotly(audio_path, sr, y, labels, cluster_regions, similarities, semantic_centroids, top_n=5)

        return smoothed_labels, reduced_embeddings, breakpoints, similarities, json_data

    # # Step 10: Full Pipeline
    # def process_audio_with_temporal_structure(audio_file_path, hop_length=320, max_clusters=10):
    #     embeddings, sr, y = extract_wav2vec_embeddings(audio_file_path)

    #     # Apply attention to embeddings
    #     embeddings = apply_attention_to_embeddings(embeddings)

    #     # Reduce dimensionality
    #     reduced_embeddings = reduce_embeddings(embeddings)

    #     # Determine optimal clusters
    #     optimal_clusters = determine_optimal_clusters(reduced_embeddings, max_clusters=max_clusters)

    #     # Perform clustering
    #     labels = cluster_embeddings(reduced_embeddings, n_clusters=optimal_clusters)

    #     # Apply temporal smoothing
    #     smoothed_labels = smooth_labels(labels)

    #     # Detect change points
    #     breakpoints = detect_change_points(reduced_embeddings)

    #     # Visualize clusters and transitions
    #     visualize_clusters_and_transitions(audio_file_path, sr, y, smoothed_labels, breakpoints, hop_length)

    #     return smoothed_labels, reduced_embeddings, breakpoints

    # Run the pipeline
    # audio_path = "sound-seq.wav"  # Replace with your audio file path
    labels, embeddings, transitions, similarities, json_data = process_audio_with_clap_comparison(audio_path)
    return json_data
# labels, embeddings, transitions = process_audio_with_temporal_structure(audio_path)
# objectifier("temp_seq-sine-sq-saw-noise-filt-1-3-5khz-pianoTriad-pianoCluster.wav")
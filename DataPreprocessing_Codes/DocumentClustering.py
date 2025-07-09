import os
import shutil
import re
import numpy as np
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_documents_with_hdbscan(folder_path, min_cluster_size=5, min_samples=None, reduce_dim=True, n_components=50, visualize=True):
    """
    Cluster text files in a folder using semantic embeddings + HDBSCAN.

    Args:
        folder_path (str): Path to folder containing `.txt` files.
        min_cluster_size (int): Minimum number of points in a cluster.
        min_samples (int): Number of samples in a neighborhood for a point to be a core point.
        reduce_dim (bool): Whether to reduce embedding dimensions using UMAP.
        n_components (int): Number of dimensions to reduce to (if reduce_dim is True).
        visualize (bool): Whether to plot cluster visualization.

    Returns:
        dict: Mapping of cluster_id -> list of filenames in that cluster.
    """
    # Validate path
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    # Step 1: Load text files
    texts, filenames = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    texts.append(f.read())
            filenames.append(file)

    if not texts:
        raise ValueError("No `.txt` files found in the directory.")

    # Step 2: Encode text to embeddings
    print("🔍 Generating document embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Step 3: Reduce dimensions with UMAP
    if reduce_dim:
        print("📉 Reducing dimensions with UMAP...")
        reducer = umap.UMAP(
            n_components=min(n_components, len(texts) - 1),
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine"
        )
        embeddings = reducer.fit_transform(embeddings)

    # Step 4: Apply HDBSCAN clustering
    print("📦 Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        gen_min_span_tree=True,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)

    # Step 5: Group filenames by cluster
    cluster_result = {}
    for idx, label in enumerate(labels):
        cluster_result.setdefault(label, []).append(filenames[idx])

    # Step 6: Visualize
    if visualize:
        _visualize_clusters(embeddings, labels, clusterer)

    return cluster_result


def _visualize_clusters(embeddings, labels, clusterer):
    """Plot UMAP embeddings with HDBSCAN cluster labels."""
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(unique_labels))]

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:
            plt.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c='gray', alpha=0.3, s=30, label='Noise'
            )
        else:
            plt.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[color], alpha=0.7, s=50, label=f"Cluster {label}"
            )

    plt.title(f"HDBSCAN Document Clustering ({len(unique_labels) - 1} clusters + noise)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def generate_cluster_names(cluster_dict, folder_path, top_n=3):
    """
    Generate names for clusters based on TF-IDF keywords.

    Args:
        cluster_dict: {cluster_id: [filenames]}
        folder_path: Path to directory with .txt files
        top_n: Number of keywords to use as cluster label

    Returns:
        dict: {cluster_id: "keyword1, keyword2, keyword3"}
    """
    cluster_names = {}
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    for cluster_id, files in cluster_dict.items():
        full_text = ""
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    full_text += f.read() + " "
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    full_text += f.read() + " "

        # Generate TF-IDF scores
        if not full_text.strip():
            cluster_names[cluster_id] = "Undefined"
            continue

        tfidf_matrix = vectorizer.fit_transform([full_text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top N keywords
        top_indices = tfidf_scores.argsort()[::-1][:top_n]
        top_keywords = [feature_array[i] for i in top_indices]
        cluster_names[cluster_id] = ", ".join(top_keywords)

    return cluster_names

def sanitize_folder_name(name):
    """Sanitize and convert label to lowercase with underscores."""
    name = name.lower().strip()
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)     # Remove special chars
    name = re.sub(r'\s+', '_', name)               # Replace spaces with _
    return name

def organize_documents_by_cluster(folder_path, cluster_dict, cluster_names):
    """
    Create subfolders and move .txt files into them based on cluster labels.

    Args:
        folder_path: Path to the directory containing the .txt files.
        cluster_dict: Dictionary {cluster_id: [filenames]}
        cluster_names: Dictionary {cluster_id: label_name}
    """
    for cluster_id, files in cluster_dict.items():
        label = cluster_names.get(cluster_id, f"cluster_{cluster_id}")
        folder_name = sanitize_folder_name(label)
        target_folder = os.path.join(folder_path, folder_name)

        # Create subfolder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Move files into the subfolder
        for filename in files:
            source_path = os.path.join(folder_path, filename)
            target_path = os.path.join(target_folder, filename)

            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"✅ Moved: {filename} → {folder_name}/")
            else:
                print(f"⚠️ File not found: {filename}")

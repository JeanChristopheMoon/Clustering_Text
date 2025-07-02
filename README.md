# Clustering Evaluation Metrics

✅ Internal Evaluation (No Ground Truth Required)

    Silhouette Score (Basic but reliable)
    Measures cohesion vs separation: higher is better (range -1 to 1).
    → Good for general-purpose clustering.

    Davies–Bouldin Index (More sensitive than silhouette)
    Lower = better. Penalizes overlapping clusters.

    Calinski–Harabasz Index (Good for spherical clusters)
    Higher = better. Focuses on between- vs. within-cluster dispersion.

    Dunn Index (Less common, more theoretical)
    Higher = better. Favors well-separated, compact clusters.

    Density-Based Validation (Advanced, for DBSCAN/HDBSCAN)

        Cluster Validity Index (CVI)

        Relative Validity Index (RVI)

    Topic Coherence (Text-specific, advanced)
    Used in BERTopic, LDA. Measures semantic similarity among top words in topics.
    → More aligned with human judgment of topic quality.

✅ External Evaluation (If Ground Truth Available)

    Adjusted Rand Index (ARI)
    Measures similarity between predicted and true clusters. Adjusts for chance.

    Normalized Mutual Information (NMI)
    Measures mutual dependence between true/predicted labels. Scales 0–1.

    Fowlkes–Mallows Index (FMI)
    Balance between precision and recall for clustering. 1 = perfect.

✅ Best Practices

    Use Silhouette + Coherence for text.

    For BERTopic: topic_model.get_coherence() (e.g., c_v, u_mass).

    Visualize: Combine with UMAP/TSNE plots for intuitive insight.

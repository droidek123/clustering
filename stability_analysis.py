import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from clustering_metrics import evaluate_clustering

def clustering_stability(
    clustering_fn,
    X,
    fn_params,
    n_runs=20,
    subsample_ratio=0.85,
    random_state=42
):
    """
    Analiza stabilności klasteryzacji dla funkcji clustering_fn(X, **fn_params)
    """
    rng = np.random.default_rng(random_state)

    base_labels = clustering_fn(X, **fn_params)

    ari_scores = []
    nmi_scores = []

    for _ in range(n_runs):
        indices = rng.choice(
            len(X),
            size=int(len(X) * subsample_ratio),
            replace=False
        )

        X_sub = X[indices]
        base_sub = base_labels[indices]

        labels_sub = clustering_fn(X_sub, **fn_params)

        ari_scores.append(
            adjusted_rand_score(base_sub, labels_sub)
        )
        nmi_scores.append(
            normalized_mutual_info_score(base_sub, labels_sub)
        )

    return {
        "ARI_scores": ari_scores,
        "NMI_scores": nmi_scores,
        "ARI_mean": np.mean(ari_scores),
        "ARI_std": np.std(ari_scores),
        "NMI_mean": np.mean(nmi_scores),
        "NMI_std": np.std(nmi_scores),
    }


def stability_with_metrics(
    clustering_fn,
    X,
    fn_params,
    algorithm_name,
    n_runs=20,
    subsample_ratio=0.85,
    random_state=42
):
    """
    Analiza stabilności + klasyczne metryki jakości klasteryzacji.
    """
    # Stabilność
    stability = clustering_stability(
        clustering_fn=clustering_fn,
        X=X,
        fn_params=fn_params,
        n_runs=n_runs,
        subsample_ratio=subsample_ratio,
        random_state=random_state
    )

    # Klasteryzacja bazowa
    labels = clustering_fn(X, **fn_params)

    # Metryki jakości (TWÓJ ISTNIEJĄCY KOD)
    quality = evaluate_clustering(
        X=X,
        labels=labels,
        algorithm_name=algorithm_name
    )

    return {
        "stability": stability,
        "quality": quality
    }

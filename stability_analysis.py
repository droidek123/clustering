import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering_metrics import evaluate_clustering

def is_trivial_clustering(labels):
    unique = set(labels)
    unique.discard(-1)
    return len(unique) <= 1

def clustering_stability(
    clustering_fn,
    X,
    fn_params,
    n_runs=20,
    subsample_ratio=0.85,
    random_state=42
):
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

        if is_trivial_clustering(base_sub) or is_trivial_clustering(labels_sub):
            ari = 0.0
            nmi = 0.0
        else:
            ari = adjusted_rand_score(base_sub, labels_sub)
            nmi = normalized_mutual_info_score(base_sub, labels_sub)

        ari_scores.append(ari)
        nmi_scores.append(nmi)

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
    stability = clustering_stability(
        clustering_fn=clustering_fn,
        X=X,
        fn_params=fn_params,
        n_runs=n_runs,
        subsample_ratio=subsample_ratio,
        random_state=random_state
    )

    labels = clustering_fn(X, **fn_params)

    quality = evaluate_clustering(
        X=X,
        labels=labels,
        algorithm_name=algorithm_name
    )

    return {
        "stability": stability,
        "quality": quality
    }

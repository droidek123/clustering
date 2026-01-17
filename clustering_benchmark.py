import itertools

import numpy as np
import pandas as pd

from city_data_loader import load_city_dataset
from stability_analysis import stability_with_metrics


class ClusteringBenchmark:
    def __init__(self, algorithms, n_runs=25):
        self.algorithms = algorithms
        self.n_runs = n_runs

    def run(self, csv_files):
        all_results = []

        for csv_path in csv_files:
            X, _, feature_names = load_city_dataset(csv_path)

            for algo_name, (fn, param_grid) in self.algorithms.items():
                param_names = list(param_grid.keys())
                param_values = list(param_grid.values())

                for values in itertools.product(*param_values):
                    params = dict(zip(param_names, values))

                    result = stability_with_metrics(
                        clustering_fn=fn,
                        X=X,
                        fn_params=params,
                        algorithm_name=algo_name,
                        n_runs=self.n_runs
                    )

                    stability = result["stability"]
                    quality = result["quality"]

                    row = {
                        "dataset": csv_path,
                        "n_features": len(feature_names),
                        "algorithm": algo_name,
                        **params,
                        "ARI_mean": stability["ARI_mean"],
                        "ARI_std": stability["ARI_std"],
                        "NMI_mean": stability["NMI_mean"],
                        "NMI_std": stability["NMI_std"],
                    }

                    if quality is not None:
                        row.update({
                            "silhouette": quality["silhouette"],
                            "davies_bouldin": quality["davies_bouldin"],
                            "calinski_harabasz": quality["calinski_harabasz"],
                            "n_clusters_found": quality["n_clusters"]
                        })
                    else:
                        row.update({
                            "silhouette": np.nan,
                            "davies_bouldin": np.nan,
                            "calinski_harabasz": np.nan,
                            "n_clusters_found": 0
                        })

                    all_results.append(row)

        return pd.DataFrame(all_results)

import numpy as np

from clustering_algorithms import (dbscan_clustering, hierarchical_clustering,
                                   kmeans_clustering)
from clustering_benchmark import ClusteringBenchmark
from custom_clustering import custom_clustering


def main():
    algorithms = {
        "KMeans": (
            kmeans_clustering,
            {"n_clusters": [2, 3, 4, 5, 6]}
        ),
        "Hierarchical": (
            hierarchical_clustering,
            {"n_clusters": [2, 3, 4, 5, 6]}
        ),
        "DBSCAN": (
            dbscan_clustering,
            {
                "eps": np.linspace(1.5, 3.5, 9),
                "min_samples": [2, 4, 6, 10]
            }
        ),
        "Custom (Mean Shift)": (
            custom_clustering,
            {}
        ),
    }


    csv_files = [
        "data/polish_cities_2020.csv",
        "data/polish_cities_2020_no_people.csv",
        "data/polish_cities_2020_no_salary.csv",
        "data/polish_cities_2020_no_green.csv",
        "data/polish_cities_2020_no_green_and_nurse.csv",
    ]

    benchmark = ClusteringBenchmark(
        algorithms=algorithms,
        n_runs=25
    )

    results = benchmark.run(csv_files)
    print(results)

    results.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()

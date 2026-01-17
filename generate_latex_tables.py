import numpy as np
import pandas as pd

CSV_PATH = "benchmark_results.csv"


def header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90 + "\n")


df = pd.read_csv(CSV_PATH)

header("TABLE 1: NUMBER OF EXPERIMENTS")

counts = (
    df.groupby("algorithm")
    .size()
    .rename("Liczba eksperymentów")
)

print(counts.to_frame().to_latex(
    caption="Liczba przeprowadzonych eksperymentów dla każdego algorytmu",
    label="tab:experiment_counts"
))

header("TABLE 2: FEATURE COUNTS")

features = (
    df.groupby("dataset")["n_features"]
    .first()
)

print(features.to_frame("Liczba cech").to_latex(
    caption="Liczba cech w analizowanych wariantach zbioru danych",
    label="tab:feature_counts"
))

header("TABLE 3: MEAN ARI PER ALGORITHM")

mean_ari = (
    df[df["ARI_mean"].notna()]
    .groupby("algorithm")[["ARI_mean", "ARI_std"]]
    .mean()
)

print(mean_ari.to_latex(
    float_format="%.3f",
    caption="Średnia stabilność klasteryzacji (ARI) dla algorytmów",
    label="tab:mean_ari"
))


header("TABLE 4: ARI RANGE")

ari_range = (
    df[df["ARI_mean"].notna()]
    .groupby("algorithm")["ARI_mean"]
    .agg(["min", "max", "mean"])
)

print(ari_range.to_latex(
    float_format="%.3f",
    caption="Minimalne, maksymalne i średnie wartości ARI",
    label="tab:ari_range"
))

header("TABLE 5: ARI VS NUMBER OF FEATURES")

ari_features = (
    df[df["ARI_mean"].notna()]
    .groupby(["n_features", "algorithm"])["ARI_mean"]
    .mean()
    .unstack()
    .sort_index(ascending=False)
)

print(ari_features.to_latex(
    float_format="%.3f",
    caption="Średnia stabilność (ARI) w funkcji liczby cech",
    label="tab:ari_vs_features",
    bold_rows=True
))

header("TABLE 6: MEAN QUALITY METRICS")

quality = (
    df[df["silhouette"].notna()]
    .groupby("algorithm")[[
        "silhouette", "davies_bouldin", "calinski_harabasz"
    ]]
    .mean()
)

print(quality.to_latex(
    float_format="%.3f",
    caption="Średnie metryki jakości klasteryzacji",
    label="tab:mean_quality"
))

header("TABLE 7: MEAN NUMBER OF CLUSTERS")

clusters = (
    df[df["n_clusters_found"] > 0]
    .groupby("algorithm")["n_clusters_found"]
    .mean()
)

print(clusters.to_frame("Średnia liczba klastrów").to_latex(
    float_format="%.2f",
    caption="Średnia liczba klastrów wykrytych przez algorytmy",
    label="tab:mean_clusters"
))

header("TABLE 8: DBSCAN PARAMETER SUMMARY")

dbscan_summary = (
    df[(df["algorithm"] == "DBSCAN") & (df["ARI_mean"].notna())]
    .groupby(["eps", "min_samples"])["ARI_mean"]
    .mean()
    .reset_index()
)

print(dbscan_summary.to_latex(
    index=False,
    float_format="%.3f",
    caption="Średnia stabilność (ARI) algorytmu DBSCAN dla różnych parametrów",
    label="tab:dbscan_param_summary"
))

header("TABLE 9: FAILURE RATE")

failures = (
    df.groupby("algorithm")["ARI_mean"]
    .apply(lambda x: x.isna().mean() * 100)
    .rename("Nieudane eksperymenty [\%]")
)

print(failures.to_frame().to_latex(
    float_format="%.1f",
    caption="Odsetek eksperymentów bez sensownej klasteryzacji",
    label="tab:failure_rate"
))

header("TABLE 10: SUMMARY")

summary = pd.DataFrame({
    "Algorytm": df["algorithm"].unique(),
})

summary["Średni ARI"] = summary["Algorytm"].apply(
    lambda a: df[df["algorithm"] == a]["ARI_mean"].mean()
)

summary["Średnia Silhouette"] = summary["Algorytm"].apply(
    lambda a: df[df["algorithm"] == a]["silhouette"].mean()
)

summary["Używa parametrów gęstości"] = summary["Algorytm"].apply(
    lambda a: "Tak" if a == "DBSCAN" else "Nie"
)

print(summary.to_latex(
    index=False,
    float_format="%.3f",
    caption="Zbiorcze porównanie algorytmów klasteryzacji",
    label="tab:summary"
))

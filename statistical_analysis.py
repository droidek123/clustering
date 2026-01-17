"""Moduł do analizy statystycznej przypadków / cech w zbiorze danych

Funkcje tutaj pomagają: analizować rozkład rozmiarów klastrów, porównywać rozkłady cech
między klastrami (test Kruskala-Wallisa dla cech numerycznych oraz test chi-kwadrat
dla cech kategorycznych) i zapisać wyniki do CSV.

Użycie (przykład):
python statistical_analysis.py --csv data/polish_cities_2020.csv --cluster-col cluster --output results.csv

Wymagane pakiety: pandas, numpy, scipy, statsmodels
"""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def analyze_cluster_sizes(labels: List[int]) -> dict:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    res = {
        "n_clusters": int(len(unique)),
        "total_points": int(labels.size),
        "mean_size": float(np.mean(counts)),
        "median_size": float(np.median(counts)),
        "std_size": float(np.std(counts, ddof=0)),
        "min_size": int(np.min(counts)),
        "max_size": int(np.max(counts)),
        "counts": dict(zip(map(str, unique.tolist()), counts.tolist())),
    }
    return res


def analyze_feature_distributions(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    features: Optional[List[str]] = None,
):
    """Porównaj rozkłady cech między klastrami.

    Dla cech numerycznych: wykonuje test Kruskala-Wallisa (nienormalny / robustny).
    Dla cech kategorycznych: wykonuje test chi-kwadrat na tabeli kontyngencji.

    Zwraca DataFrame z kolumnami: feature, type, stat, p_value, extra
    """
    if features is None:
        features = [c for c in df.columns if c != cluster_col]

    clusters = df[cluster_col].dropna().unique()
    results = []

    for feat in features:
        series = df[feat]
        if pd.api.types.is_numeric_dtype(series):
            groups = [series[df[cluster_col] == k].dropna().values for k in clusters]
            try:
                stat, p = stats.kruskal(*groups)
            except Exception:
                stat, p = np.nan, np.nan
            results.append({
                "feature": feat,
                "type": "numeric",
                "stat": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
            })
        else:
            contingency = pd.crosstab(df[feat], df[cluster_col])
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
            except Exception:
                chi2, p, dof = np.nan, np.nan, None
            results.append({
                "feature": feat,
                "type": "categorical",
                "stat": float(chi2) if not np.isnan(chi2) else None,
                "p_value": float(p) if not np.isnan(p) else None,
            })

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def summarize_metrics(
    metrics_dict: dict,
    algorithm_name: str = None
) -> dict:
    """Tworzy podsumowanie statystyczne dla metryk eksperymentu.
    
    metrics_dict: słownik {"metric_name": [wartość1, wartość2, ...], ...}
    
    Zwraca DataFrame z kolumnami: algorithm, metric, mean, median, std, min, max, q25, q75
    """
    results = []
    for metric_name, values in metrics_dict.items():
        values = np.asarray(values)
        results.append({
            "algorithm": algorithm_name,
            "metric": metric_name,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "n": int(len(values)),
        })
    return pd.DataFrame(results)


def compare_algorithms(
    results_list: list,
    metric_name: str = "ARI",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Porównuje pary algorytmów testem Mann-Whitney U dla danej metryki.
    
    results_list: lista (algorithm_name, wartości_metryki)
    
    Zwraca DataFrame z wynikami porównań: algo1, algo2, stat, p_value, sig
    """
    comparisons = []
    n = len(results_list)
    
    for i in range(n):
        for j in range(i+1, n):
            name1, vals1 = results_list[i]
            name2, vals2 = results_list[j]
            
            vals1 = np.asarray(vals1)
            vals2 = np.asarray(vals2)
            
            try:
                stat, p = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                sig = "Yes" if p < alpha else "No"
            except Exception:
                stat, p, sig = np.nan, np.nan, "Error"
            
            comparisons.append({
                "algorithm_1": name1,
                "algorithm_2": name2,
                "metric": metric_name,
                "statistic": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "significant": sig,
            })
    
    return pd.DataFrame(comparisons)


def compare_algorithms_wilcoxon_holm(
    results_list: list,
    metric_name: str = "ARI",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Porównuje pary algorytmów testem Wilcoxona z poprawką Holma.
    
    Test Wilcoxona to test nieparametryczny dla par zależnych.
    Poprawka Holma kontroluje FWER (family-wise error rate) dla wielokrotnych testów.
    
    results_list: lista (algorithm_name, wartości_metryki)
    alpha: poziom istotności
    
    Zwraca DataFrame z wynikami: algo1, algo2, stat, p_value_raw, p_value_holm, significant
    """
    comparisons = []
    p_values_raw = []
    algo_pairs = []
    
    n = len(results_list)
    
    # Zbieranie statystyk dla wszystkich par
    for i in range(n):
        for j in range(i+1, n):
            name1, vals1 = results_list[i]
            name2, vals2 = results_list[j]
            
            vals1 = np.asarray(vals1)
            vals2 = np.asarray(vals2)
            
            try:
                # Test Wilcoxona (zakładamy pary obserwacji)
                stat, p = stats.wilcoxon(vals1, vals2, alternative='two-sided')
            except Exception:
                stat, p = np.nan, np.nan
            
            p_values_raw.append(p if not np.isnan(p) else 1.0)
            algo_pairs.append((name1, name2, stat))
    
    # Poprawka Holma
    if p_values_raw:
        reject, p_holm, _, _ = multipletests(p_values_raw, alpha=alpha, method='holm')
    else:
        reject, p_holm = [], []
    
    # Budowanie wyników
    for idx, (name1, name2, stat) in enumerate(algo_pairs):
        comparisons.append({
            "algorithm_1": name1,
            "algorithm_2": name2,
            "metric": metric_name,
            "statistic": float(stat) if not np.isnan(stat) else None,
            "p_value_raw": float(p_values_raw[idx]),
            "p_value_holm": float(p_holm[idx]) if idx < len(p_holm) else None,
            "significant": "Yes" if (idx < len(reject) and reject[idx]) else "No",
        })
    
    return pd.DataFrame(comparisons)


def rank_algorithms(summary_combined):
    """
    Tworzy ranking algorytmów na podstawie średnich metryk.
    Wyższe wartości = lepsze dla ARI, NMI, Silhouette, Calinski-Harabasz.
    Niższe wartości = lepsze dla Davies-Bouldin.
    
    Oczekuje summary_combined DataFrame z kolumnami:
    algorithm, metric, mean, median, std, min, max, q25, q75, n
    Gdzie każda metryka to osobny wiersz.
    """
    # Pivot: każda metryka staje się kolumną z wartością 'mean'
    pivot_df = summary_combined.pivot(index='algorithm', columns='metric', values='mean').reset_index()
    
    # Przygotuj kolumny dla normalizacji (rename dla konsystencji)
    metric_cols = ['ARI', 'NMI', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
    
    # Sprawdź jakie metryki są dostępne
    available_metrics = [m for m in metric_cols if m in pivot_df.columns]
    
    ranking = pivot_df[['algorithm'] + available_metrics].copy()
    
    # Normalizacja do 0-1 (min-max) dla każdej metryki
    normalized = ranking[available_metrics].copy()
    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val - min_val != 0:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5  # Jeśli wszystkie wartości są równe
    
    # Davies-Bouldin odwrotnie (niższe = lepsze)
    if 'Davies-Bouldin' in normalized.columns:
        normalized['Davies-Bouldin'] = 1 - normalized['Davies-Bouldin']
    
    # Średnia wszystkich dostępnych metryk = score
    ranking['score'] = normalized[available_metrics].mean(axis=1)
    
    # Sortuj i dodaj rank
    ranking = ranking.sort_values('score', ascending=False).reset_index(drop=True)
    ranking['rank'] = range(1, len(ranking) + 1)
    
    # Przygotuj kolumny wyjściowe
    output_cols = ['rank', 'algorithm', 'score']
    for metric in available_metrics:
        output_cols.append(f'mean_{metric.replace("-", "_")}')
    
    # Dodaj mean kolumny
    for metric in available_metrics:
        ranking[f'mean_{metric.replace("-", "_")}'] = pivot_df.set_index('algorithm').loc[ranking['algorithm'], metric].values
    
    return ranking[output_cols]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analiza statystyczna cech względem kolumny klastrowej")
    parser.add_argument("--csv", required=True, help="Ścieżka do pliku CSV ze zbioru danych")
    parser.add_argument("--cluster-col", default="cluster", help="Nazwa kolumny z etykietami klastrów")
    parser.add_argument("--features", default=None, help="Lista cech rozdzielona przecinkami (domyślnie wszystkie poza kolumną klastrów)")
    parser.add_argument("--output", default="stat_analysis_results.csv", help="Plik wyjściowy CSV z wynikami")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = None

    if args.cluster_col not in df.columns:
        raise SystemExit(f"Kolumna klastrów '{args.cluster_col}' nie istnieje w {args.csv}")

    results_df = analyze_feature_distributions(df, cluster_col=args.cluster_col, features=features)
    save_results(results_df, args.output)
    print(f"Zapisano wyniki do {args.output}")

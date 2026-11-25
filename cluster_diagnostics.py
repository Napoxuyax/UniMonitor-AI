"""Оценка качества и устойчивости кластеров за выбранный год."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from threadpoolctl import threadpool_limits
from core import FEATURES


def evaluate_clusters(frame, year=None, max_clusters=8, repeats=5):
    year = frame.year.max() if year is None else year
    sample = frame.loc[frame.year == year].copy().reset_index(drop=True)
    if len(sample) < 4 or not 2 <= max_clusters <= 20 or not 2 <= repeats <= 30:
        raise ValueError('Недостаточно наблюдений или неверные параметры')
    x = StandardScaler().fit_transform(sample[FEATURES])
    unique = len(np.unique(x, axis=0))
    ceiling = min(max_clusters, len(x)-1, unique)
    rows, labels = [], {}
    with threadpool_limits(limits=1):
        for k in range(2, ceiling+1):
            reference = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
            if not 2 <= len(np.unique(reference)) < len(x):
                continue
            alternatives = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(x) for seed in range(repeats)]
            rows.append({'clusters':k, 'silhouette':silhouette_score(x,reference),
                         'davies_bouldin':davies_bouldin_score(x,reference),
                         'calinski_harabasz':calinski_harabasz_score(x,reference),
                         'stability_ari':float(np.mean([adjusted_rand_score(reference,a) for a in alternatives]))})
            labels[k] = reference
    if not rows:
        raise ValueError('Недостаточно различных профилей вузов')
    metrics = pd.DataFrame(rows).sort_values(['silhouette','clusters'],ascending=[False,True]).reset_index(drop=True)
    best = int(metrics.iloc[0].clusters)
    sample['cluster'] = labels[best]+1
    profiles = sample.groupby('cluster')[FEATURES].mean()
    profiles['count'] = sample.groupby('cluster').size()
    return metrics, sample, profiles.reset_index()

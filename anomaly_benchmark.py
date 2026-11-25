"""Temporal benchmark with explicitly injected artificial anomalies.
The source observations are treated as reference, not verified normal cases.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from core import FEATURES

ARCHITECTURES = [(8, 2, 8), (16, 3, 16), (16, 8, 2, 8, 16)]


def inject(frame, seed=42, fraction=.25):
    """Perturb one feature per selected record; preserve physical limits."""
    if not 0 < fraction <= 1 or len(frame) < 4:
        raise ValueError('Некорректный объём контрольных аномалий')
    rng = np.random.default_rng(seed)
    altered = frame.copy().reset_index(drop=True)
    indices = rng.choice(len(altered), max(1, round(len(altered)*fraction)), replace=False)
    altered = altered.iloc[indices].copy().reset_index(drop=True)
    changed = []
    for i in altered.index:
        column = str(rng.choice(FEATURES))
        before = float(altered.at[i, column])
        if column in ('score', 'employment'):
            after = before-25 if before>=50 else before+25
        else:
            after = before*2.5 if before else 1.
        altered[column] = altered[column].astype(float)
        altered.at[i, column] = after
        changed.append(column)
    original = frame.copy().reset_index(drop=True).assign(injected=0, changed_feature='')
    altered = altered.assign(injected=1, changed_feature=changed)
    return pd.concat([original, altered], ignore_index=True)


def benchmark(df, quantile=.95):
    if not 0 < quantile < 1:
        raise ValueError('Некорректный квантиль')
    years = sorted(df.year.unique())
    if len(years)<4:
        raise ValueError('Необходимы минимум четыре года наблюдений')
    train = df[df.year < years[-2]]
    calibration = df[df.year == years[-2]]
    test = df[df.year == years[-1]]
    if min(len(train), len(calibration), len(test)) < 4:
        raise ValueError('Недостаточно наблюдений')
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES])
    c = scaler.transform(calibration[FEATURES])
    validation = inject(calibration, seed=41)
    evaluation = inject(test, seed=42)
    v = scaler.transform(validation[FEATURES])
    t = scaler.transform(evaluation[FEATURES])
    methods = {}
    for layers in ARCHITECTURES:
        model = MLPRegressor(hidden_layer_sizes=layers, solver='lbfgs',
                             alpha=.1, max_iter=3000, max_fun=50000, random_state=42)
        model.fit(x,x)
        methods['AE '+str(layers)] = lambda a, m=model: np.mean((a-m.predict(a))**2,axis=1)
    forest = IsolationForest(n_estimators=150, random_state=42).fit(x)
    methods['Isolation Forest'] = lambda a: -forest.score_samples(a)
    metrics, predictions = [], []
    for name, score in methods.items():
        threshold = float(np.quantile(score(c),quantile))
        valid_scores = score(v)
        test_scores = score(t)
        flags = test_scores > threshold
        y = evaluation.injected
        metrics.append(dict(model=name, threshold=threshold,
                            validation_ap=average_precision_score(validation.injected,valid_scores),
                            test_ap=average_precision_score(y,test_scores),
                            roc_auc=roc_auc_score(y,test_scores),
                            precision=precision_score(y,flags,zero_division=0),
                            recall=recall_score(y,flags,zero_division=0),
                            f1=f1_score(y,flags,zero_division=0)))
        predictions.append(evaluation[['university','year','injected','changed_feature']].assign(
            model=name, anomaly_score=test_scores, flagged=flags))
    results = pd.DataFrame(metrics).sort_values('validation_ap',ascending=False)
    return results, pd.concat(predictions,ignore_index=True), dict(train=len(train), calibration=len(calibration), test=len(test), injected=int(evaluation.injected.sum()), calibration_year=int(years[-2]), test_year=int(years[-1]), selected=str(results.iloc[0].model))


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from core import load_data
    parser = argparse.ArgumentParser(description='Контрольный эксперимент обнаружения искусственных аномалий')
    parser.add_argument('data')
    parser.add_argument('--output',default='anomaly_results')
    args=parser.parse_args()
    metrics,predictions,manifest=benchmark(load_data(args.data))
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(output/'metrics.csv',index=False)
    predictions.to_csv(output/'predictions.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False))

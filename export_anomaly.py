"""Persist benchmark detectors with their preprocessing and thresholds."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from anomaly_benchmark import ARCHITECTURES, inject, benchmark
from core import FEATURES, load_data
from model_registry import digest


def export_detectors(dataset, output):
    frame = load_data(dataset)
    metrics, predictions, meta = benchmark(frame)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = frame[frame.year < meta['calibration_year']]
    calibration = frame[frame.year == meta['calibration_year']]
    test = inject(frame[frame.year == meta['test_year']], seed=42)
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES])
    c = scaler.transform(calibration[FEATURES])
    t = scaler.transform(test[FEATURES])
    models = []
    for number, layers in enumerate(ARCHITECTURES, 3):
        model = MLPRegressor(hidden_layer_sizes=layers, solver='lbfgs', alpha=.1,
                             max_iter=3000, max_fun=50000, random_state=42).fit(x,x)
        models.append((number, 'AE '+str(layers), model))
    models.append((6, 'Isolation Forest', IsolationForest(n_estimators=150, random_state=42).fit(x)))
    records = []
    for number, name, model in models:
        def score(estimator, values):
            if name == 'Isolation Forest':
                return -estimator.score_samples(values)
            return np.mean((values-estimator.predict(values))**2, axis=1)
        threshold = float(np.quantile(score(model,c), .95))
        filename = f'model{number}.joblib'
        joblib.dump(dict(model=model, scaler=scaler, threshold=threshold, features=FEATURES, name=name), output/filename)
        restored = joblib.load(output/filename)
        actual = score(restored['model'], restored['scaler'].transform(test[FEATURES]))
        expected = predictions.loc[predictions.model==name,'anomaly_score'].to_numpy()
        np.testing.assert_allclose(actual,expected)
        records.append(dict(file=filename,name=name,threshold=threshold))
    for name, table in [('train',train),('calibration',calibration),('test',test),('metrics',metrics),('predictions',predictions)]:
        table.to_csv(output/f'{name}.csv',index=False)
    meta.update(models=records,dataset_sha256=digest(dataset),kind='demo' if Path(dataset).name=='demo.csv' else 'external_unverified',files={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset'); p.add_argument('output'); a=p.parse_args()
    print(json.dumps(export_detectors(a.dataset,a.output),ensure_ascii=False,indent=2))

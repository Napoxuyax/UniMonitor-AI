"""Analysis of published monitoring data; zero scores are ambiguous, not imputed."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

FEATURES = ['students','score','research','income']
ROOT = Path(__file__).parent

def load_panel(path=ROOT/'real_data/panel.csv'):
    frame = pd.read_csv(path, dtype={'id':str})
    if frame.duplicated(['id','year']).any():
        raise ValueError('Повтор идентификатора и года')
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError('Неполные или неконечные показатели')
    if (frame[FEATURES]<0).any().any() or (frame.score>100).any():
        raise ValueError('Недопустимое значение показателя')
    return frame

def annual_quality(frame):
    return frame.assign(zero_score=frame.score.eq(0),positive_score=frame.score.gt(0),employment_available=frame.employment.notna()).groupby('year').agg(rows=('id','size'),zero_score=('zero_score','sum'),positive_score=('positive_score','sum'),employment_available=('employment_available','sum')).reset_index()

def transform(frame):
    x = frame[FEATURES].astype(float).copy()
    for col in ('students','research','income'):
        x[col] = np.log1p(x[col])
    return x

def inject(frame, seed):
    rng = np.random.default_rng(seed)
    altered = frame.sample(frac=.25,random_state=seed).copy()
    for idx in altered.index:
        col = rng.choice(FEATURES)
        value = altered.at[idx,col]
        altered.at[idx,col] = (value-25 if value>=50 else value+25) if col=='score' else (value*3 if value else 1)
    return pd.concat([frame.assign(injected=0),altered.assign(injected=1)],ignore_index=True)

def benchmark_actual(frame,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    quality=annual_quality(frame); quality.to_csv(output/'quality_by_year.csv',index=False)
    eligible=frame[frame.score.gt(0)].copy()
    train=eligible[eligible.year<2024]; cal=eligible[eligible.year.eq(2024)]; test=eligible[eligible.year.eq(2025)]
    if min(len(train),len(cal),len(test))<4:
        raise ValueError('Недостаточно данных для временного разделения')
    scaler=RobustScaler().fit(transform(train)); x=scaler.transform(transform(train)); c=scaler.transform(transform(cal))
    validation=inject(cal,41); evaluation=inject(test,42)
    v=scaler.transform(transform(validation)); t=scaler.transform(transform(evaluation))
    rows=[]; predictions=[]
    models=[('AE 8-2-8',MLPRegressor(hidden_layer_sizes=(8,2,8),max_iter=400,random_state=42,alpha=.1,tol=.0005)),('AE 16-3-16',MLPRegressor(hidden_layer_sizes=(16,3,16),max_iter=400,random_state=42,alpha=.1,tol=.0005)),('Isolation Forest',IsolationForest(n_estimators=100,random_state=42))]
    for i,(name,model) in enumerate(models,1):
        ae=name.startswith('AE')
        if ae: model.fit(x,x)
        else: model.fit(x)
        def score(a): return np.mean((a-model.predict(a))**2,axis=1) if ae else -model.score_samples(a)
        threshold=float(np.quantile(score(c),.95)); s=score(t); flags=s>threshold; y=evaluation.injected
        rows.append(dict(model=name,threshold=threshold,validation_ap=average_precision_score(validation.injected,score(v)),test_ap=average_precision_score(y,s),roc_auc=roc_auc_score(y,s),precision=precision_score(y,flags,zero_division=0),recall=recall_score(y,flags,zero_division=0),f1=f1_score(y,flags,zero_division=0)))
        predictions.append(evaluation[['id','year','injected']].assign(model=name,anomaly_score=s,flagged=flags))
        file=output/f'model{i}.joblib'; joblib.dump(dict(model=model,scaler=scaler,threshold=threshold,features=FEATURES),file)
        restored=joblib.load(file); z=restored['scaler'].transform(transform(evaluation)); m=restored['model']; np.testing.assert_allclose(s,np.mean((z-m.predict(z))**2,axis=1) if ae else -m.score_samples(z))
    metrics=pd.DataFrame(rows).sort_values('validation_ap',ascending=False); metrics.to_csv(output/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(output/'predictions.csv',index=False)
    for name,part in [('train',train),('calibration',cal),('test',evaluation)]: part.to_csv(output/f'{name}.csv',index=False)
    manifest=dict(train=len(train),calibration=len(cal),test=len(test),injected=int(evaluation.injected.sum()),selected=metrics.iloc[0].model,zero_policy='Исключены нулевые баллы ЕГЭ из эксперимента, исходные данные сохранены',label_policy='Искусственные отклонения на реальных исходных наблюдениях; истинные аномалии не размечены')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics,manifest

if __name__=='__main__':
    print(benchmark_actual(load_panel(),ROOT/'actual_anomalies'))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
FEATURES=['students','score','research','income','employment']
KEY=['university','region','year']

def load_data(source):
    df=pd.read_csv(source)
    missing=set(KEY+FEATURES)-set(df.columns)
    if missing: raise ValueError('Отсутствуют поля: '+', '.join(sorted(missing)))
    df=df[KEY+FEATURES].copy()
    if df.empty: raise ValueError('Нет наблюдений')
    for col in KEY[:2]:
        df[col]=df[col].astype('string').str.strip()
        if df[col].isna().any() or df[col].eq('').any(): raise ValueError('Пустое название или регион')
    for col in ['year']+FEATURES:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if not np.isfinite(df[['year']+FEATURES]).all().all(): raise ValueError('Некорректные числовые значения')
    if (df[FEATURES]<0).any().any() or not df.employment.between(0,100).all() or not df.score.between(0,100).all(): raise ValueError('Показатели вне допустимого диапазона')
    if not df.year.between(1900,2100).all() or (df.year%1!=0).any(): raise ValueError('Некорректный год')
    if df.duplicated(['university','year']).any(): raise ValueError('Повторяющиеся записи вуз–год')
    return df.sort_values(['university','year']).reset_index(drop=True)

def analyze(df):
    latest=df[df.year==df.year.max()].copy()
    if len(latest)<4: raise ValueError('Для анализа нужны минимум четыре вуза за последний год')
    x=StandardScaler().fit_transform(latest[FEATURES])
    model=KMeans(n_clusters=min(3,len(latest)),random_state=42,n_init=10)
    latest['cluster']=model.fit_predict(x)+1
    ae=MLPRegressor(hidden_layer_sizes=(8,2,8),random_state=42,max_iter=1000,solver='lbfgs')
    ae.fit(x,x)
    latest['anomaly_score']=np.mean((x-ae.predict(x))**2,axis=1)
    latest['anomaly']=latest.anomaly_score>latest.anomaly_score.quantile(.9)
    return latest

def forecast(df):
    pairs=df.copy()
    pairs['next_year']=pairs.groupby('university').year.shift(-1)
    pairs['target']=pairs.groupby('university').employment.shift(-1)
    pairs=pairs[pairs.next_year==pairs.year+1].dropna(subset=['target'])
    years=sorted(pairs.next_year.unique())
    if len(years)<3: raise ValueError('Нужны минимум три последовательных целевых года')
    train=pairs[pairs.next_year<years[-1]]; test=pairs[pairs.next_year==years[-1]]
    scaler=StandardScaler().fit(train[FEATURES]); a=scaler.transform(train[FEATURES]); b=scaler.transform(test[FEATURES])
    models={'MLP':TransformedTargetRegressor(regressor=MLPRegressor(hidden_layer_sizes=(16,8),solver='lbfgs',max_iter=2500,max_fun=30000,random_state=42), transformer=StandardScaler()),'Ridge':Ridge()}
    metrics={}
    for name,model in models.items():
        model.fit(a,train.target)
        pred=np.clip(model.predict(b),0,100)
        metrics[name]={'MAE':float(mean_absolute_error(test.target,pred)),'RMSE':float(np.sqrt(mean_squared_error(test.target,pred)))}
    metrics['Наивный прогноз']={'MAE':float(mean_absolute_error(test.target,test.employment)),'RMSE':float(np.sqrt(mean_squared_error(test.target,test.employment)))}
    latest=df[df.year==df.year.max()].copy()
    production_scaler=StandardScaler().fit(pairs[FEATURES])
    production_model=models['MLP']
    production_model.fit(production_scaler.transform(pairs[FEATURES]),pairs.target)
    latest['forecast_employment']=np.clip(production_model.predict(production_scaler.transform(latest[FEATURES])),0,100)
    return metrics,latest[['university','year','employment','forecast_employment']]

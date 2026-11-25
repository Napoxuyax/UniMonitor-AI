"""Temporal forecasting on actual monitoring records, with a separate historical employment experiment."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run(panel, output, target, features, validation_years, test_year):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(panel,dtype={'id':str}).sort_values(['id','year'])
    g=d.groupby('id'); d['target']=g[target].shift(-1); d['target_year']=g.year.shift(-1)
    d=d[d.target_year.eq(d.year+1)].dropna(subset=features+['target'])
    d=d[d.target_year<=test_year]
    configs={f'Ridge {a}':make_pipeline(StandardScaler(),Ridge(alpha=a)) for a in (.1,1,10)}
    for h in [(16,), (32,16),(32,16,8)]:
        configs[f'MLP {h}']=TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=h,solver='adam',max_iter=500,early_stopping=False,random_state=42,alpha=.1,tol=.0005)),transformer=StandardScaler())
    def metrics(y,p):
        return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
    def predict(m,x): return np.clip(m.predict(x),0,100 if target in ('score','employment') else np.inf)
    rows=[]
    for name,m in configs.items():
        for year in validation_years:
            train=d[d.target_year<year]; val=d[d.target_year==year]
            m.fit(train[features],train.target)
            rows.append({'model':name,'year':year,**metrics(val.target,predict(m,val[features]))})
    folds=pd.DataFrame(rows); folds.to_csv(output/'folds.csv',index=False)
    ranking=folds.groupby('model').MAE.mean().sort_values(); ranking.to_csv(output/'ranking.csv')
    train=d[d.target_year<test_year]; test=d[d.target_year==test_year]; train.to_csv(output/'train.csv',index=False); test.to_csv(output/'test.csv',index=False)
    pred=test[['id','name','year','target_year','target']].copy(); rows=[]
    for i,prefix in enumerate(('MLP','Ridge'),1):
        name=next(n for n in ranking.index if n.startswith(prefix)); m=configs[name]; m.fit(train[features],train.target)
        p=predict(m,test[features]); file=output/f'model{i}.joblib'; joblib.dump(m,file); np.testing.assert_allclose(p,predict(joblib.load(file),test[features]))
        pred[f'model{i}']=p; rows.append({'model':name,**metrics(test.target,p)})
    rows.append({'model':'Наивный прогноз',**metrics(test.target,test[target])}); pred.to_csv(output/'predictions.csv',index=False)
    result=pd.DataFrame(rows); result.to_csv(output/'comparison.csv',index=False)
    meta={'target':target,'features':features,'validation_years':validation_years,'test_year':test_year,'train_rows':len(train),'test_rows':len(test),'winner_validation':ranking.index[0],'year_definition':'Год мониторинга, сведения за предыдущий год','metrics':rows}
    (output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':
    run('real_data/panel.csv','real_score',['score'][0],['students','score','research','income'],[2022,2023,2024],2025)
    run('real_data/panel.csv','real_employment','employment',['students','score','research','income','employment'],[2017],2018)

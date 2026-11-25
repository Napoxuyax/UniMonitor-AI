from pathlib import Path
import pytest
from core import load_data,analyze,forecast
DATA=Path(__file__).resolve().parents[1]/'data/demo.csv'
def test_load():
    d=load_data(DATA); assert len(d)==420; assert d.university.nunique()==60
def test_analysis():
    d=analyze(load_data(DATA)); assert len(d)==60; assert d.cluster.nunique()==3; assert d.anomaly_score.ge(0).all()
def test_forecast():
    m,d=forecast(load_data(DATA)); assert set(m)=={'MLP','Ridge','Наивный прогноз'}; assert d.forecast_employment.between(0,100).all()
def test_duplicates(tmp_path):
    import pandas as pd
    d=load_data(DATA); p=tmp_path/'bad.csv'; pd.concat([d,d.iloc[:1]]).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Повторяющиеся'): load_data(p)

from actual_monitoring import load_panel
from actual_dashboard import select_rows, statistics, actual_flags
from test_secure_web import server, client
from urllib.parse import urlencode
import pandas as pd
from io import BytesIO
import pytest

def test_actual_statistics():
    rows=select_rows(load_panel(),2025)
    stats=statistics(rows)
    assert len(rows)==1251
    assert stats.loc[1,'Наблюдений']==1085
    assert stats.loc[0,'Наблюдений']==1251

def test_actual_flags():
    rows=select_rows(load_panel(),2025)
    flags=actual_flags(rows)
    assert len(flags)==1085 and flags.anomaly_score.ge(0).all()
    with pytest.raises(ValueError): actual_flags(select_rows(load_panel(),2023))

def test_actual_filter_download(server):
    c,b=client(server,'viewer')
    query=urlencode({'year':2025,'region':'Республика Адыгея'})
    result=c.open(b+'/actual-download?'+query)
    df=pd.read_csv(BytesIO(result.read()))
    assert len(df)>0 and df.region.eq('Республика Адыгея').all() and df.year.eq(2025).all()
    assert 'Карточка организации' in c.open(b+'/actual-monitoring?'+query).read().decode()

def test_actual_history(server):
    c,b=client(server,'viewer')
    text=c.open(b+'/actual-monitoring?year=2020&id=1').read().decode()
    assert 'Показать историю' in text and 'Нейросетевая диагностика' not in text

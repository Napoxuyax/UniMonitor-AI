from actual_monitoring import load_panel, annual_quality, transform, inject, FEATURES
from test_secure_web import server, client
import numpy as np

def test_actual_quality():
    data=load_panel(); q=annual_quality(data)
    assert len(data)==13956
    assert q.loc[q.year.eq(2025),'zero_score'].iloc[0]==166
    assert not q.loc[q.year.gt(2018),'employment_available'].any()

def test_transform_and_injection():
    data=load_panel().query('year == 2025 and score > 0')
    augmented=inject(data,42)
    assert augmented.injected.sum()==271
    assert np.isfinite(transform(augmented)).all().all()
    assert list(transform(augmented).columns)==FEATURES

def test_actual_page(server):
    c,b=client(server,'viewer')
    text=c.open(b+'/actual-monitoring?year=2025').read().decode()
    assert 'Фактический мониторинг вузов' in text and 'tochno.st' in text and '166' in text

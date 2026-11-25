from pathlib import Path
import pytest
import numpy as np
from core import load_data
from scenario import scenario_analysis

@pytest.fixture
def data():
    return load_data(Path(__file__).parents[1]/'data/demo.csv')

def test_zero(data):
    rows, imp, meta=scenario_analysis(data)
    assert np.allclose(rows.delta,0)
    assert len(imp)==5
    assert meta['test_year']==2025

def test_bounds(data):
    rows, _, meta=scenario_analysis(data,{'employment':100})
    assert rows.lower.between(0,100).all()
    assert rows.upper.between(0,100).all()
    assert (rows.lower<=rows.scenario).all()
    assert (rows.upper>=rows.scenario).all()
    assert 0<=meta['observed_test_coverage']<=1

def test_invalid(data):
    with pytest.raises(ValueError): scenario_analysis(data,{'income':101})
    with pytest.raises(ValueError): scenario_analysis(data,{'unknown':1})
    with pytest.raises(ValueError): scenario_analysis(data,coverage=1)

def test_no_control_leakage(data):
    _, _, a=scenario_analysis(data)
    changed=data.copy(); changed.loc[changed.year==2025,'employment']=0
    _, _, b=scenario_analysis(changed)
    assert a['radius']==b['radius']

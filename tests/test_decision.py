from pathlib import Path
import numpy as np
import pytest
from core import load_data, FEATURES
from decision import normalized_weights, rank_universities, sensitivity, distribution_drift

@pytest.fixture
def data():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')

def test_weights():
    assert np.isclose(normalized_weights().sum(),1)
    with pytest.raises(ValueError): normalized_weights(dict.fromkeys(FEATURES,0))
    with pytest.raises(ValueError): normalized_weights(dict.fromkeys(FEATURES,float('nan')))

def test_rank(data):
    r=rank_universities(data,2025)
    assert len(r)==60 and r['rank'].iloc[0]==1
    assert r.score_total.between(0,100).all()
    assert r.score_total.is_monotonic_decreasing

def test_one_weight(data):
    w={f:float(f=='employment') for f in FEATURES}
    r=rank_universities(data,2025,w)
    assert r.university.iloc[0] in set(data.loc[(data.year==2025)&(data.employment==data.loc[data.year==2025,'employment'].max()),'university'])

def test_sensitivity(data):
    r=sensitivity(data,2025,repetitions=30)
    assert (r.rank_best<=r.rank_median).all() and (r.rank_median<=r.rank_worst).all()
    assert r.equals(sensitivity(data,2025,repetitions=30))
    assert r.top10_share.between(0,100).all()

def test_drift(data):
    r=distribution_drift(data,2019,2025)
    assert len(r)==5 and (r.matched==60).all()
    assert r.ks_distance.between(0,1).all()
    with pytest.raises(ValueError): distribution_drift(data,2025,2019)

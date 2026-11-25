from pathlib import Path
import pandas as pd
import pytest
from core import load_data
from university_profile import profile, export

@pytest.fixture
def data():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')

def test_historical_profile(data):
    name=data.university.iloc[0]
    a=profile(data,name,2024)
    b=profile(data[data.year<=2024],name,2024)
    for key in ('history','rank','peers','anomalies','trends'):
        pd.testing.assert_frame_equal(a[key],b[key])
    assert a['history'].year.max()==2024
    assert len(a['peers'])==4

def test_small_sample(data):
    name=data.university.iloc[0]
    a=profile(data[data.university.eq(name)],name,2025,forecast=True)
    assert a['peers'].empty and a['rank'].empty
    assert a['manifest']['status']['Сравнение'].startswith('Недостаточно')

def test_missing_year(data):
    with pytest.raises(ValueError): profile(data,data.university.iloc[0],1900)

def test_export(data,tmp_path):
    r=profile(data,data.university.iloc[0],2025)
    export(r,tmp_path)
    assert (tmp_path/'manifest.json').is_file()
    assert len(pd.read_csv(tmp_path/'peers.csv'))==4

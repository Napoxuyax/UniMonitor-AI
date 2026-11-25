from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json
import pandas as pd
import pytest
from core import load_data
from reporting import build_report, diagnostics
ROOT=Path(__file__).resolve().parents[1]

def test_export_archive():
    df=load_data(ROOT/'data/demo.csv')
    html, blob, manifest=build_report(df,'<script>','demo')
    assert b'&lt;script&gt;' in html
    with ZipFile(BytesIO(blob)) as z:
        assert z.testzip() is None
        assert len(pd.read_csv(z.open('normalized.csv')))==420
        assert json.load(z.open('manifest.json'))==manifest

def test_diagnostics():
    df=load_data(ROOT/'data/demo.csv')
    p=pd.read_csv(ROOT/'research_results/predictions.csv')
    overall, regions, errors=diagnostics(p,df)
    assert overall['n']==60
    assert regions.n.sum()==60
    assert overall['MAE']==pytest.approx(p.error.abs().mean())
    assert errors.absolute_error.is_monotonic_decreasing

def test_reject_invalid_prediction():
    df=load_data(ROOT/'data/demo.csv')
    p=pd.read_csv(ROOT/'research_results/predictions.csv')
    p.loc[0,'prediction']=101
    with pytest.raises(ValueError): diagnostics(p,df)

def test_reject_unknown_source_type():
    df=load_data(ROOT/'data/demo.csv')
    with pytest.raises(ValueError): build_report(df,'test','unknown')

def test_export_control_results():
    df=load_data(ROOT/'data/demo.csv')
    result={'predictions':pd.read_csv(ROOT/'research_results/predictions.csv'),'comparison':pd.read_csv(ROOT/'research_results/comparison.csv'),'winner':'Ridge','test_year':2025}
    _,blob,manifest=build_report(df,'demo','demo',result)
    assert manifest['diagnostics']['n']==60
    with ZipFile(BytesIO(blob)) as z: assert 'regional_errors.csv' in z.namelist()

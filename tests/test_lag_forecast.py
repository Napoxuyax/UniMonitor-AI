from pathlib import Path
import numpy as np
from core import load_data
from lag_forecast import windows, run


def frame():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')


def test_windows_no_gap():
    df = frame()
    name = df.university.iloc[0]
    df = df[~((df.university==name)&(df.year==2023))]
    rows, columns = windows(df)
    assert len(columns)==15
    assert not ((rows.university==name)&rows.year.isin([2023,2024,2025])).any()


def test_window_targets():
    df = frame()
    rows, _ = windows(df)
    row = rows.iloc[0]
    original = df[(df.university==row.university)&(df.year==row.target_year)]
    assert row.target == original.employment.iloc[0]
    assert rows[rows.year==2025].target.isna().all()


def test_experiment():
    results = run(frame())
    assert results['manifest']['test_year']==2025
    assert max(results['folds'].year)<2025
    assert len(results['comparison'])==4
    assert len(results['forecast'])==60
    assert np.isfinite(results['comparison'].MAE).all()
    assert results['forecast'].forecast.between(0,100).all()

from pathlib import Path
from core import load_data
from research import build_pairs, scores, experiment
import numpy as np

DATA=Path(__file__).resolve().parents[1]/'data/demo.csv'

def test_pairs_are_consecutive():
    pairs=build_pairs(load_data(DATA))
    assert (pairs.target_year==pairs.year+1).all()
    assert len(pairs)==360

def test_scores_perfect():
    assert scores(np.array([10,20]),np.array([10,20]))=={'MAE':0.,'RMSE':0.,'R2':1.}

def test_temporal_experiment():
    result=experiment(load_data(DATA))
    assert result['test_year']==2025
    assert result['folds'].year.max()<2025
    assert result['folds'].model.nunique()==9
    assert len(result['predictions'])==60
    assert result['forecast'].forecast_year.eq(2026).all()
    assert result['forecast'].forecast_employment.between(0,100).all()

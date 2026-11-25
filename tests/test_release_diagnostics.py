from pathlib import Path
import pytest
from core import load_data
from release_diagnostics import panel_quality, rolling_evaluation


def frame():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')


def test_coverage():
    df = frame()
    name = df.university.iloc[0]
    df = df[~((df.university==name)&(df.year==2023))]
    coverage, gaps = panel_quality(df)
    assert coverage.loc[coverage.year==2023,'observed'].iloc[0]==59
    assert gaps.loc[gaps.university==name,'internal_gaps'].iloc[0]==1


def test_rolling_no_future():
    folds, summary, predictions = rolling_evaluation(frame())
    assert (folds.max_train_year < folds.year).all()
    assert len(summary)==3
    assert predictions.prediction.between(0,100).all()
    assert not folds[folds.model=='Наивный прогноз'].improved.any()


def test_future_perturbation():
    df = frame()
    before, _, _ = rolling_evaluation(df)
    df.loc[df.year==2025,'employment']=0
    after, _, _ = rolling_evaluation(df)
    assert before[before.year<2025].equals(after[after.year<2025])


def test_insufficient():
    with pytest.raises(ValueError):
        rolling_evaluation(frame().query('year>=2024'))

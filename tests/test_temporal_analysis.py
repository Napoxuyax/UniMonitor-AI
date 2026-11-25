from pathlib import Path
import pandas as pd
import pytest
from core import load_data, FEATURES
from temporal_analysis import annual_changes, detect_temporal_anomalies, trend_profiles, anomaly_summary


def fixture_frame():
    frame = pd.DataFrame({'university':'A','region':'R','year':range(2018,2025)})
    for metric in FEATURES:
        frame[metric] = [10,11,13,16,18,21,60]
    return frame


def test_jump():
    result = detect_temporal_anomalies(fixture_frame())
    assert result[result.year.eq(2024)].anomaly.all()


def test_no_future_leakage():
    frame = fixture_frame()
    first = detect_temporal_anomalies(frame)
    other = detect_temporal_anomalies(frame[frame.year<2024])
    pd.testing.assert_frame_equal(first[first.year<2024].reset_index(drop=True),other.reset_index(drop=True))


def test_gap_and_zero():
    frame = fixture_frame()
    frame.loc[0,'students']=0
    frame=frame[frame.year.ne(2020)]
    result=annual_changes(frame)
    assert result[result.year.eq(2021)].delta.isna().all()
    assert result[(result.year.eq(2019)) & (result.metric.eq('students'))].relative_change.isna().all()
    assert trend_profiles(frame).missing_years.eq(1).all()


def test_demo_summary():
    frame=load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')
    summary=anomaly_summary(detect_temporal_anomalies(frame))
    assert len(summary)==5
    assert summary.anomalies.le(summary.assessed).all()
    assert len(trend_profiles(frame))==300


def test_invalid_threshold():
    with pytest.raises(ValueError): detect_temporal_anomalies(fixture_frame(),0)


def test_zero_mad_not_scored():
    frame=fixture_frame()
    for metric in FEATURES:
        frame[metric]=[10,12,14,16,18,20,60]
    result=detect_temporal_anomalies(frame)
    last=result[result.year.eq(2024)]
    assert last.status.eq('Нулевая вариативность истории').all()
    assert last.robust_score.isna().all()
    assert not last.anomaly.any()

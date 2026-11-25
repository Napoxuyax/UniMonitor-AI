from pathlib import Path
import pytest
from core import load_data
from sensitivity import analyze_sensitivity


def frame():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')


def test_shapes_and_bounds():
    details, summary, ranges, manifest = analyze_sensitivity(frame())
    assert len(details) == 600 and len(summary) == 10 and len(ranges) == 5
    assert details.forecast.between(0,100).all()
    assert summary.outside_share.between(0,1).all()
    assert manifest['train_count'] == 360


def test_no_future():
    df = frame()
    before = analyze_sensitivity(df, 2024)[0]
    df.loc[df.year==2025,'employment'] = 0
    after = analyze_sensitivity(df, 2024)[0]
    assert before.equals(after)


def test_outside_detection():
    df = frame()
    df.loc[df.year==2025,'students'] = 10000000
    details, _, _, manifest = analyze_sensitivity(df)
    assert manifest['baseline_outside_count'] == 60
    assert details.scenario_outside.all()


@pytest.mark.parametrize('step',[0,51,float('nan')])
def test_invalid_step(step):
    with pytest.raises(ValueError):
        analyze_sensitivity(frame(), step=step)

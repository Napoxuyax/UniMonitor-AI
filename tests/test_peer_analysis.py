from pathlib import Path
import pytest
from core import load_data
from peer_analysis import run


def frame():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')


def test_excludes_self_and_other_years():
    r=run(frame(),2024,5)
    m=r['memberships']
    assert (m.university!=m.peer).all()
    assert m.year.eq(2024).all()
    assert m.groupby('university').size().eq(5).all()


def test_nearest_sizes():
    df=frame()
    r=run(df,2025,10)
    x=r['memberships']
    import numpy as np
    latest=df[df.year.eq(2025)].set_index('university')
    for name, group in x.groupby('university'):
        expected=(np.log(latest.students)-np.log(latest.loc[name,'students'])).abs().drop(name).sort_values().head(10)
        assert set(group.peer)==set(expected.index)


def test_metrics_and_advice():
    r=run(frame())
    c=r['comparisons']
    assert len(c)==240
    assert c.peer_percentile.between(0,100).all()
    assert (c.peer_q1<=c.peer_median).all()
    assert (c.peer_median<=c.peer_q3).all()
    assert r['recommendations'].university.nunique()==60


def test_invalid_group():
    with pytest.raises(ValueError):
        run(frame(),2025,2)
    with pytest.raises(ValueError):
        run(frame(),1900,10)

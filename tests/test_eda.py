from pathlib import Path
from core import load_data
from eda import explore
import pytest

DATA=Path(__file__).resolve().parents[1]/'data/demo.csv'

def test_coverage():
    result=explore(load_data(DATA))
    assert result['coverage'].observations.sum()==420
    assert result['quality']['universities']==60

def test_correlation():
    result=explore(load_data(DATA))
    assert result['correlation'].shape==(5,5)
    assert result['correlation'].equals(result['correlation'].T)

def test_gap_not_treated_as_annual_change():
    df=load_data(DATA)
    df=df[df.year.ne(2020)]
    changes=explore(df)['changes']
    assert changes.loc[changes.year.eq(2021),'employment_change'].isna().all()

def test_empty():
    with pytest.raises(ValueError): explore(load_data(DATA).iloc[:0])

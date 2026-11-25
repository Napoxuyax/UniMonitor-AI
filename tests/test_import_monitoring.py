import pandas as pd
import pytest
from import_monitoring import prepare, FIELDS

MAP={x:x for x in FIELDS}
def row(**changes):
    return dict(university='Вуз А',region='Москва',year=2025,students=100,score='75,5',research=20,income=50,employment=80)|changes

def test_decimal_and_year():
    a,r,q=prepare(pd.DataFrame([row()]),MAP,'Учебный пример',-1)
    assert a.iloc[0].score==75.5 and a.iloc[0].year==2024 and r.empty
    assert q['accepted_rows']==1

def test_reject_invalid():
    a,r,q=prepare(pd.DataFrame([row(),row(university='Б',employment=None),row(university='В',score=101)]),MAP,'Учебный пример')
    assert len(a)==1 and len(r)==2 and q['missing_by_field']['employment']==1

def test_duplicate_rejected():
    a,r,q=prepare(pd.DataFrame([row(),row()]),MAP,'Учебный пример')
    assert a.empty and len(r)==2

def test_mapping_required():
    with pytest.raises(ValueError): prepare(pd.DataFrame([row()]),{},'Учебный пример')

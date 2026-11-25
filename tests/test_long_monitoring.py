import pandas as pd
import pytest
from core import FEATURES
from long_monitoring import convert


def fixture():
    config = {'columns':{'university':'name','region':'area','year':'period','indicator':'code','value':'val'},'indicators':{f:{'code':str(i),'unit':'test unit','definition':'test definition'} for i,f in enumerate(FEATURES)}}
    rows = [{'name':'Test A','area':'R','period':2025,'code':str(i),'val':str(v)} for i,v in enumerate([1000,70,30,100,80])]
    return pd.DataFrame(rows), config


def test_long_format():
    raw, config = fixture()
    accepted, rejected, conflicts, report = convert(raw,config,'fixture')
    assert len(accepted)==1 and accepted.employment.iloc[0]==80
    assert rejected.empty and conflicts.empty and report['selected_long_rows']==5


def test_duplicate_not_averaged():
    raw, config = fixture()
    raw = pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    accepted,rejected,conflicts,report = convert(raw,config,'fixture')
    assert accepted.empty and len(rejected)==1 and len(conflicts)==2


def test_missing_indicator_not_imputed():
    raw,config=fixture()
    accepted,rejected,_,_=convert(raw.iloc[:-1],config,'fixture')
    assert accepted.empty and len(rejected)==1


def test_definition_required():
    raw,config=fixture()
    config['indicators']['employment']['definition']=''
    with pytest.raises(ValueError):
        convert(raw,config,'fixture')

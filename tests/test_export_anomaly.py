import json
from pathlib import Path
import joblib
from export_anomaly import export_detectors
from model_registry import digest


def test_export_detectors(tmp_path):
    output = tmp_path / 'detectors'
    meta = export_detectors(Path(__file__).resolve().parents[1]/'data/demo.csv', output)
    assert len(meta['models']) == 4
    assert meta['kind'] == 'demo'
    for name, checksum in meta['files'].items():
        assert digest(output/name) == checksum
    for record in meta['models']:
        model = joblib.load(output/record['file'])
        assert model['threshold'] == record['threshold']
        assert len(model['features']) == 5

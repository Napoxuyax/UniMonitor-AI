from pathlib import Path
import pytest
from core import load_data
from cluster_diagnostics import evaluate_clusters
from operations import TrainingQueue
from storage import Store


def test_cluster_metrics():
    frame = load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')
    metrics, sample, profiles = evaluate_clusters(frame)
    assert len(sample)==60 and profiles['count'].sum()==60
    assert metrics.silhouette.between(-1,1).all()
    assert metrics.silhouette.is_monotonic_decreasing
    assert sample.cluster.nunique()==metrics.iloc[0].clusters


def test_cluster_invalid():
    frame = load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')
    with pytest.raises(ValueError):
        evaluate_clusters(frame.head(1))


def test_persistent_history(tmp_path):
    class Registry:
        def train(self,*args): return {'id':'example'}
    store=Store(tmp_path/'db.sqlite')
    frame=load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')
    q=TrainingQueue(Registry(),store=store)
    job=q.submit('admin',frame,'demo','demo','Ridge')
    q.close()
    q=TrainingQueue(Registry(),store=store)
    assert q.list()[0]['id']==job and q.list()[0]['status']=='completed'
    q.close()

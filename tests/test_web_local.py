from threading import Thread
from urllib.request import urlopen
from urllib.error import HTTPError
from io import BytesIO
from zipfile import ZipFile
import pytest
from web_local import create_server
@pytest.fixture(scope='module')
def base():
    server=create_server(0); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown(); server.server_close(); thread.join()
@pytest.mark.parametrize('route,text',[('/','420'),('/compare?year=2024&feature=score','Первые 10'),('/eda','Корреляции Спирмена'),('/anomalies','Результаты анализа'),('/reports','Скачать архив')])
def test_pages(base,route,text):
    with urlopen(base+route) as r: assert text in r.read().decode()
def test_export(base):
    with urlopen(base+'/download') as r: data=r.read()
    with ZipFile(BytesIO(data)) as z: assert 'manifest.json' in z.namelist()
def test_invalid(base):
    with pytest.raises(HTTPError) as exc: urlopen(base+'/compare?feature=bad')
    assert exc.value.code==400

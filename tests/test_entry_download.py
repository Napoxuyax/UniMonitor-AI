import io
from email.message import Message
import pytest
import download_dataset as module
from main import main


class Response(io.BytesIO):
    def __init__(self, data, mime='text/csv'):
        super().__init__(data)
        self.headers = Message()
        self.headers['Content-Type'] = mime

    def geturl(self):
        return 'https://example.org/data.csv'


def test_download(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: Response(b'a,b\n1,2\n'))
    path = tmp_path / 'data.csv'
    result = module.download('https://example.org/data.csv', path)
    assert result['bytes'] == len(path.read_bytes())
    assert len(result['sha256']) == 64
    with pytest.raises(FileExistsError):
        module.download('https://example.org/data.csv', path)


@pytest.mark.parametrize('data,mime,limit', [(b'', 'text/csv', 10), (b'abc', 'text/html', 10), (b'abc', 'text/csv', 2)])
def test_invalid_download(tmp_path, monkeypatch, data, mime, limit):
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: Response(data, mime))
    path = tmp_path / 'data.csv'
    with pytest.raises(ValueError):
        module.download('https://example.org/data.csv', path, limit)
    assert not path.exists()


def test_https_only(tmp_path):
    with pytest.raises(ValueError):
        module.download('http://example.org/data.csv', tmp_path / 'data.csv')


def test_main_help(capsys):
    with pytest.raises(SystemExit) as result:
        main(['--help'])
    assert result.value.code == 0
    assert '--init-admin' in capsys.readouterr().out

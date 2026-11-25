from threading import Thread
from urllib.request import build_opener, HTTPCookieProcessor
from urllib.error import HTTPError
from urllib.parse import urlencode
from http.cookiejar import CookieJar
import re
import pytest
from secure_web import create_secure_server

@pytest.fixture(scope='module')
def server(tmp_path_factory):
    folder=tmp_path_factory.mktemp('secure')
    s=create_secure_server(0,folder/'db.sqlite',folder/'models')
    s.store.bootstrap('admin','password-admin-123')
    s.store.create_user('admin','viewer','password-viewer-123','viewer')
    t=Thread(target=s.serve_forever,daemon=True); t.start()
    yield s, f'http://127.0.0.1:{s.server_port}'
    s.shutdown();s.server_close();t.join()

def client(server,role=None):
    s,base=server
    c=build_opener(HTTPCookieProcessor(CookieJar()))
    if role: c.open(base+'/login',urlencode({'login':role,'password':f'password-{role}-123'}).encode()).read()
    return c,base

def post(c,base,path,**fields):
    return c.open(base+path,urlencode(fields).encode()).read().decode()

def csrf(c,base):
    return re.search(r'name="csrf" value="([^"]+)"',c.open(base+'/datasets').read().decode())[1]

def test_guest(server):
    c,b=client(server); assert c.open(b+'/compare').geturl().endswith('/login')

def test_wrong_password(server):
    c,b=client(server)
    with pytest.raises(HTTPError) as e: post(c,b,'/login',login='admin',password='wrong')
    assert e.value.code==401

def test_admin(server):
    c,b=client(server,'admin'); assert 'Администрирование' in c.open(b+'/admin').read().decode()

def test_viewer_admin(server):
    c,b=client(server,'viewer')
    with pytest.raises(HTTPError) as e: c.open(b+'/admin')
    assert e.value.code==403

def test_csrf(server):
    c,b=client(server,'admin')
    with pytest.raises(HTTPError) as e: post(c,b,'/select',dataset=0)
    assert e.value.code==403

def test_import_select_isolation(server):
    s,b=server;c,b=client(server,'admin'); other,_=client(server,'viewer')
    frame=s.frame.head(4)
    post(c,b,'/import',csrf=csrf(c,b),name='Test subset',source='test source',kind='demo',csv=frame.to_csv(index=False))
    dataset=s.store.list('admin')[0]['id']
    post(c,b,'/select',csrf=csrf(c,b),dataset=dataset)
    assert 'test source' in c.open(b+'/').read().decode()
    assert '420' in other.open(b+'/').read().decode()

def test_viewer_import(server):
    s,b=server;c,b=client(server,'viewer')
    with pytest.raises(HTTPError) as e: post(c,b,'/import',csrf=csrf(c,b),name='x',source='x',kind='demo',csv=s.frame.to_csv(index=False))
    assert e.value.code==403

def test_logout(server):
    c,b=client(server,'admin');post(c,b,'/logout',csrf=csrf(c,b));assert c.open(b+'/').geturl().endswith('/login')

@pytest.mark.parametrize('route,expected',[('/ranking','Устойчивость места'),('/drift','KS-расстояние')])
def test_decision_pages(server,route,expected):
    c,b=client(server,'viewer')
    assert expected in c.open(b+route).read().decode()

def test_invalid_weights(server):
    c,b=client(server,'viewer')
    with pytest.raises(HTTPError) as e: c.open(b+'/ranking?students=-1')
    assert e.value.code==400


def test_experiments_admin(server):
    c,b=client(server,"admin")
    assert "Реестр экспериментов" in c.open(b+"/experiments").read().decode()

def test_experiments_viewer(server):
    c,b=client(server,"viewer")
    with pytest.raises(HTTPError) as e: c.open(b+"/experiments")
    assert e.value.code==403


def test_temporal_page(server):
    c,b=client(server,'viewer')
    assert 'Сводка диагностики' in c.open(b+'/temporal').read().decode()


def test_lag_page(server):
    c,b=client(server,'viewer')
    assert 'Прогноз по истории' in c.open(b+'/lag-forecast').read().decode()

def test_profile_page(server):
    c,b=client(server,'viewer')
    page=c.open(b+'/profile?year=2024').read().decode()
    assert 'Карточка вуза' in page and 'Устойчивость рейтинга' in page
    with pytest.raises(HTTPError) as e: c.open(b+'/profile?year=1900')
    assert e.value.code==400


def test_diagnostics_page(server):
    c,b=client(server,'viewer')
    assert 'Устойчивость прогнозов' in c.open(b+'/diagnostics?run=1').read().decode()


def test_sensitivity_page(server):
    c,b=client(server,'viewer')
    assert 'Чувствительность прогнозов' in c.open(b+'/sensitivity?step=5').read().decode()


def test_relogin_invalidates_old_session(server):
    s,b=server
    c,b=client(server,'admin')
    old=set(s.sessions)
    post(c,b,'/login',login='admin',password='password-admin-123')
    assert len(old-set(s.sessions)) == 1
    assert 'Администрирование' in c.open(b+'/admin').read().decode()


def test_http_throttle(server):
    from auth_guard import LoginGuard
    s,b=server
    original=s.login_guard
    s.login_guard=LoginGuard(attempts=2)
    try:
        c,b=client(server)
        for _ in range(2):
            with pytest.raises(HTTPError) as e:
                post(c,b,'/login',login='admin',password='wrong')
            assert e.value.code==401
        with pytest.raises(HTTPError) as e:
            post(c,b,'/login',login='admin',password='wrong')
        assert e.value.code==429
    finally:
        s.login_guard=original

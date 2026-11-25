"""Render authenticated HTTP responses; not a browser navigation test."""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.request import build_opener, HTTPCookieProcessor
from urllib.parse import urlencode
from http.cookiejar import CookieJar
from playwright.sync_api import sync_playwright
from secure_web import create_secure_server


def main():
    output = Path(__file__).parent / 'actual_screenshots'
    output.mkdir(exist_ok=True)
    with TemporaryDirectory() as directory:
        server = create_secure_server(0, Path(directory)/'test.sqlite', Path(directory)/'models')
        server.store.bootstrap('render', 'Temporary-render-12345')
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            client = build_opener(HTTPCookieProcessor(CookieJar()))
            client.open(base+'/login', urlencode({'login':'render','password':'Temporary-render-12345'}).encode()).read()
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
                page=browser.new_page(viewport={'width':1600,'height':1000})
                routes=[('overview',{}),('region',{'year':2025,'region':'Республика Адыгея'}),('history',{'year':2020,'id':'1'})]
                for name,query in routes:
                    with client.open(base+'/actual-monitoring?'+urlencode(query)) as response:
                        content=response.read().decode()
                    page.set_content(content)
                    assert page.locator('h1').inner_text()=='Фактический мониторинг вузов'
                    page.screenshot(path=output/(name+'.png'))
                    if name=='overview':
                        page.get_by_role('heading',name='Нейросетевая диагностика',exact=True).scroll_into_view_if_needed()
                        page.screenshot(path=output/'anomalies.png')
                        page.get_by_role('heading',name='Прогноз ЕГЭ: эксперимент с исходными нулями',exact=True).scroll_into_view_if_needed()
                        page.screenshot(path=output/'forecast.png')
                    if name=='history':
                        page.get_by_role('heading',name='Карточка организации',exact=True).scroll_into_view_if_needed()
                        page.screenshot(path=output/'history_detail.png')
                    print(name+': HTTP response rendered')
                browser.close()
        finally:
            server.shutdown(); server.server_close(); worker.join()

if __name__=='__main__':
    main()

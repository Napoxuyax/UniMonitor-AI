"""Browser verification of the actual monitoring dashboard with a temporary account."""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from playwright.sync_api import sync_playwright
from secure_web import create_secure_server


def main():
    output = Path(__file__).parent / 'actual_screenshots'
    output.mkdir(exist_ok=True)
    with TemporaryDirectory() as folder:
        server = create_secure_server(0, Path(folder)/'db.sqlite', Path(folder)/'models')
        server.store.bootstrap('admin', 'Temporary-demo-12345')
        server.store.create_user('admin', 'viewer', 'Temporary-viewer-12345', 'viewer')
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path='/usr/bin/chromium', args=['--no-sandbox','--disable-dev-shm-usage','--no-proxy-server'])
                page = browser.new_page(viewport={'width': 1600, 'height': 1000}, accept_downloads=True)
                page.goto(base+'/login',timeout=15000)
                page.locator('input[name=login]').fill('viewer')
                page.locator('input[name=password]').fill('Temporary-viewer-12345')
                page.get_by_role('button',name='Войти').click()
                page.goto(base+'/actual-monitoring')
                assert page.locator('h1').inner_text() == 'Фактический мониторинг вузов'
                page.screenshot(path=output/'01_actual_overview.png')
                print('Actual dashboard: OK',flush=True)
                page.locator('select[name=region]').select_option('Республика Адыгея')
                page.get_by_role('button',name='Применить').click()
                assert 'Республика Адыгея' in page.url or 'region=' in page.url
                page.screenshot(path=output/'02_region.png')
                print('Region filter: OK',flush=True)
                with page.expect_download() as d:
                    page.get_by_text('Скачать выбранные данные').click()
                assert d.value.suggested_filename=='actual-monitoring.csv'
                print('CSV download: OK',flush=True)
                page.goto(base+'/actual-monitoring?year=2020&id=1')
                assert page.get_by_role('heading',name='Карточка организации').count()==1
                page.get_by_role('heading',name='Карточка организации').scroll_into_view_if_needed()
                page.screenshot(path=output/'03_history.png')
                print('Historical card: OK',flush=True)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

if __name__=='__main__':
    main()

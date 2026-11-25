"""Проверка локального интерфейса в Chromium (опционально: playwright)."""
from threading import Thread
from pathlib import Path
from playwright.sync_api import sync_playwright
from web_local import create_server

def main():
    output=Path(__file__).parent/'screenshots'; output.mkdir(exist_ok=True)
    server=create_server(0); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
            for route,name,heading in [('/','01_overview','Обзор'),('/compare','02_compare','Сравнение вузов'),('/anomalies','03_anomalies','Кластеры и аномалии'),('/eda','04_eda','Исследование данных'),('/forecast?run=1','05_forecast','Прогнозирование'),('/reports','06_reports','Отчёты')]:
                response=page.goto(base+route,timeout=60000); assert response.status==200; assert page.locator('h1').inner_text()==heading
                page.screenshot(path=output/(name+'.png'),full_page=True); print(name+': OK')
            page.goto(base+'/compare'); page.select_option('select[name=year]','2024'); page.select_option('select[name=feature]','score'); page.get_by_role('button',name='Сравнить').click(); assert 'year=2024' in page.url; print('Фильтры: OK')
            page.goto(base+'/reports')
            with page.expect_download() as d: page.get_by_text('Скачать архив отчёта').click()
            assert d.value.suggested_filename=='report.zip'; print('Скачивание: OK')
            browser.close()
    finally: server.shutdown(); server.server_close(); thread.join()
if __name__=='__main__': main()

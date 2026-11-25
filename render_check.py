"""Снимки HTML-ответов локального сервера через Chromium без сетевой навигации."""
from threading import Thread
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright
from web_local import create_server

def main():
    server=create_server(0); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    output=Path(__file__).parent/'screenshots'; output.mkdir(exist_ok=True)
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':1000})
            for route,name in [('/','01_overview'),('/compare','02_compare'),('/anomalies','03_anomalies'),('/eda','04_eda'),('/forecast?run=1','05_forecast'),('/reports','06_reports')]:
                with urlopen(f'http://127.0.0.1:{server.server_port}'+route) as response: html=response.read().decode()
                page.set_content(html); assert page.locator('h1').count()==1
                page.screenshot(path=output/(name+'.png'),full_page=True); print(name+': HTML rendered')
            browser.close()
    finally: server.shutdown(); server.server_close(); thread.join()
if __name__=='__main__': main()

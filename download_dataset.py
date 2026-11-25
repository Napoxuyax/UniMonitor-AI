"""Загрузка внешнего CSV по заданной оператором ссылке (только CLI)."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


def download(url, destination, max_bytes=512 * 1024 * 1024):
    # Сетевой запрос доступен оператору через CLI, не посетителям веб-сервера.
    if urlsplit(url).scheme != 'https' or not urlsplit(url).hostname:
        raise ValueError('Требуется прямая HTTPS-ссылка')
    if max_bytes <= 0:
        raise ValueError('Лимит размера должен быть положительным')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    checksum = hashlib.sha256()
    size = 0
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temp = Path(directory) / 'download'
        with urlopen(url, timeout=30) as response, temp.open('wb') as out:
            final_url = response.geturl()
            if urlsplit(final_url).scheme != 'https':
                raise ValueError('Небезопасное перенаправление')
            if response.headers.get_content_type() == 'text/html':
                raise ValueError('Получена веб-страница вместо файла')
            for chunk in iter(lambda: response.read(65536), b''):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError('Превышен лимит размера файла')
                checksum.update(chunk)
                out.write(chunk)
        if not size:
            raise ValueError('Получен пустой файл')
        # Не перезаписываем ранее загруженный файл даже при конкурентном запуске.
        import os
        os.link(temp, destination)
    return {'url': url, 'final_url': final_url, 'bytes': size, 'sha256': checksum.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description='Загрузка исходного набора данных')
    parser.add_argument('url')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = download(args.url, args.destination)
    manifest = args.destination.with_suffix(args.destination.suffix + '.source.json')
    with manifest.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

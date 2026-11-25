"""Единая точка входа локального приложения UniMonitor AI."""
import argparse
import getpass
from pathlib import Path
from secure_web import create_secure_server
from storage import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description='UniMonitor AI: локальная система мониторинга вузов')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--database', default='unimonitor.db')
    parser.add_argument('--registry', default=str(Path(__file__).parent / 'model_runs'))
    parser.add_argument('--init-admin', action='store_true')
    args = parser.parse_args(argv)
    if args.init_admin:
        login = input('Логин: ')
        password = getpass.getpass('Пароль (12+ символов): ')
        print('Создан:', Store(args.database).bootstrap(login, password))
        return
    # По умолчанию сервер доступен только на локальном компьютере.
    server = create_secure_server(args.port, args.database, args.registry)
    print(f'http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

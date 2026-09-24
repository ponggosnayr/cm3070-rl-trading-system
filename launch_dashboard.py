"""Local dashboard launcher. Owns and stops only the processes it starts."""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.backend:
        sys.path.insert(0, str(ROOT/'env/Lib/site-packages'))
        import uvicorn
        from api import app
        uvicorn.run(app, host='127.0.0.1', port=8000)
        return

    node = shutil.which('node')
    vite = ROOT/'frontend/node_modules/vite/bin/vite.js'
    if not node or not vite.exists():
        raise RuntimeError('Node.js or frontend dependencies are missing. See README.md.')
    for port in (8000, 5173):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is already in use. Close the previous app launcher and try again.')

    log_dir = ROOT/'logs'
    log_dir.mkdir(exist_ok=True)
    children, handles = [], []
    try:
        commands = [([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--backend'], ROOT),
                    ([node, str(vite), '--host', '127.0.0.1', '--port', '5173', '--strictPort'], ROOT/'frontend')]
        for name, (command, cwd) in zip(('backend','frontend'), commands):
            log = open(log_dir/f'launcher_{name}.log', 'w', encoding='utf-8')
            handles.append(log)
            children.append(subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW))
        print('Starting your dashboard...', flush=True)
        pending = {'http://127.0.0.1:8000/', 'http://127.0.0.1:5173/'}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic()+60
        while pending and time.monotonic()<deadline:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('A server could not start. Check logs/launcher_backend.log and launcher_frontend.log.')
            for url in list(pending):
                try:
                    with opener.open(url,timeout=1) as response:
                        if response.status==200:
                            pending.remove(url)
                except (OSError, urllib.error.URLError):
                    pass
            if pending:
                time.sleep(.25)
        if pending:
            raise RuntimeError('Startup timed out. Check the launcher logs in the logs folder.')
        print('Ready: http://localhost:5173', flush=True)
        if not args.smoke:
            webbrowser.open('http://localhost:5173')
            input('Keep this window open. Press Enter here to stop the app.\n')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Could not launch: {exc}', file=sys.stderr)
        sys.exit(1)

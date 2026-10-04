"""One command to run the whole app locally:  uv run python run.py   (add --no-browser to skip opening a tab)"""
import argparse
import threading
import time
import webbrowser

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(f"http://{a.host}:{a.port}")), daemon=True).start()
    uvicorn.run("src.api.main:app", host=a.host, port=a.port, log_level="warning")

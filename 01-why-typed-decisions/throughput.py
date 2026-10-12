"""How many typed decisions a second the server gives, with 1, 2, 4 and 8 requests at once.

    python 01-why-typed-decisions/throughput.py
    python 01-why-typed-decisions/throughput.py --n 100     # messages per level

Each level sends --n BANKING77 test messages (different ones per level) through a pool of that
many threads, each with its own connection, and reports requests per second and the time each
request took. The request is the typed one from typed.py. The first step fills the server's 8
slots with the system prompt, so no level pays for reading it.
"""

import argparse
import http.client
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from data import load
from prompt import build
from server import URL  # importing server checks that the server is up

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

url = urlsplit(URL)
ds = load("banking77")
prompt = build(ds.taxonomy, ds.shots())
texts = [text for _, text, _ in ds.split("test")]
local = threading.local()  # one connection per thread


def send(body: dict) -> dict:
    if not hasattr(local, "conn"):
        local.conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=300)
    local.conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    return json.loads(local.conn.getresponse().read())


def decide(text: str) -> float:
    """One typed decision; returns its time in ms."""
    body = {"messages": [{"role": "system", "content": prompt.system}, {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 2, "grammar": prompt.grammar,
            "logprobs": True, "top_logprobs": 30}
    start = time.perf_counter()
    try:
        send(body)
    except (http.client.RemoteDisconnected, ConnectionError):
        # The server closes a connection that has sat idle for a few seconds; open a new one.
        local.conn.close()
        del local.conn
        start = time.perf_counter()
        send(body)
    return (time.perf_counter() - start) * 1000


start = time.perf_counter()
with ThreadPoolExecutor(8) as pool:
    list(pool.map(decide, texts[:8]))
print(f"Filled 8 slots with the system prompt in {time.perf_counter() - start:.0f} s.\n")

print("at once   requests/s   median ms   95th pct ms")
for k, at_once in enumerate([1, 2, 4, 8]):
    batch = texts[8 + k * args.n: 8 + (k + 1) * args.n]
    with ThreadPoolExecutor(at_once) as pool:
        start = time.perf_counter()
        ms = sorted(pool.map(decide, batch))
        wall = time.perf_counter() - start
    p95 = ms[max(0, round(0.95 * len(ms)) - 1)]
    print(f"{at_once:7}   {len(ms) / wall:10.2f}   {statistics.median(ms):9.0f}   {p95:11.0f}")

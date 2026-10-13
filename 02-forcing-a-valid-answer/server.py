"""Talk to llama-server (started by 00-setup/start-server.sh).

The address comes from LLAMA_URL (default http://127.0.0.1:8080). One connection is kept open and
reused, so requests don't pay for a new connection each time. Importing this module checks that
llama-server is answering there, and exits with a message if it isn't. Standard library only.
"""

import http.client
import json
import os
import sys
from urllib.parse import urlsplit

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
_url = urlsplit(URL)
_conn = http.client.HTTPConnection(_url.hostname, _url.port or 80, timeout=300)


def check() -> None:
    """Exit with a message unless llama-server is up and has loaded its model."""
    conn = http.client.HTTPConnection(_url.hostname, _url.port or 80, timeout=10)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        status, body = resp.status, resp.read()
    except OSError:
        sys.exit(f"No server at {URL}. Start it with 00-setup/start-server.sh, "
                 "or set LLAMA_URL if it runs elsewhere.")
    except http.client.HTTPException:
        status, body = None, b""
    finally:
        conn.close()
    if status == 503:
        sys.exit(f"llama-server at {URL} is still loading the model. Try again in a moment.")
    if status != 200 or b'"ok"' not in body:
        sys.exit(f"Something at {URL} answered, but it isn't llama-server. "
                 "Set LLAMA_URL to the server's address.")


def post(path: str, body: dict) -> dict:
    _conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = _conn.getresponse()
    data = json.loads(resp.read())
    if resp.status != 200:
        raise RuntimeError(f"{path}: HTTP {resp.status}: {data}")
    return data


def chat(system: str, user: str, **params) -> dict:
    """One chat completion. params go into the request as they are (grammar, max_tokens, ...)."""
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, **params}
    return post("/v1/chat/completions", body)


def tokenize(text: str) -> list[str]:
    """How the model's tokenizer splits text: one string per token."""
    return [t["piece"] for t in post("/tokenize", {"content": text, "with_pieces": True})["tokens"]]


check()

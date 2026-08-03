# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A local stand-in for the orders API the stage-0 tools call.

Stage 0 runs with no external service. Point ORDERS_API_BASE at this stub and
the three tools return payloads instead of DNS failures.

The order and return payloads are imported from the Gateway Lambda rather than
copied, so stage 0 and stage 1 return byte-identical results by construction.
That identity is what makes the stage-1 verification a real before-and-after
instead of two runs that merely look similar.

Run it standalone:

    python -m examples.stage0_langgraph.local_api        # serves on port 8080

or borrow it in-process, on an ephemeral port:

    with running_stub() as base_url:
        os.environ["ORDERS_API_BASE"] = base_url
"""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from examples.gateway.lambda_target.lambda_function import lookup_order, process_return


def search_faq(query: str) -> dict:
    """The FAQ answer. This tool has no Lambda twin: it never moves to Gateway."""
    return {
        "query": query,
        "answer": (
            "Items can be returned within 30 days of delivery in their original "
            "packaging. Refunds post 3-5 business days after we receive the item."
        ),
        "article_id": "FAQ-RETURNS-001",
        "source": "local knowledge base",
    }


class _Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/orders/"):
            self._send(lookup_order(url.path[len("/orders/") :]))
        elif url.path == "/faq/search":
            self._send(search_faq(parse_qs(url.query).get("q", [""])[0]))
        else:
            self._send({"error": f"No such path: {url.path}"}, status=404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "Body is not JSON"}, status=400)
            return
        if url.path == "/returns":
            self._send(process_return(args.get("order_id", ""), args.get("reason", "")))
        else:
            self._send({"error": f"No such path: {url.path}"}, status=404)

    def log_message(self, *args) -> None:
        """Silence the per-request stderr log so it does not bury the CLI output."""


@contextmanager
def running_stub(port: int = 0):
    """Serve the stub on a background thread, yielding its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8080), _Handler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Orders API stub on {base_url} — Ctrl-C to stop")
    print(f"  export ORDERS_API_BASE={base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()

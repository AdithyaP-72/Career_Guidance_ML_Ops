"""Abuse protection for the public API: request-size limits and rate limits.

The service is public and runs on 0.1 CPU, so one script could make it unusable for everyone. Two guards:

BodySizeLimit   rejects a request body that is too big BEFORE it is read: by its Content-Length header, and by
                counting bytes for bodies sent without one. (FastAPI otherwise receives a whole upload, however
                big, before our own 5 MB check runs.)
RateLimiter     sliding one-minute window, per visitor AND in total. The per-visitor key comes from headers set by
                the proxy in front of Render; a client can fake those, so the total limit is the backstop that
                still holds when someone sends a new fake address with every request.

Counters live in memory: the API runs one worker process, so there is one place to count. They reset on restart.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable

from fastapi import HTTPException, Request


class BodySizeLimit:
    """Pure ASGI middleware: 413 for request bodies over the limit (per path, else `default`)."""

    def __init__(self, app, default: int, per_path: dict[str, int] | None = None):
        self.app, self.default, self.per_path = app, default, per_path or {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.per_path.get(scope["path"], self.default)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            return await _reply_413(send, limit)

        received = 0

        async def counted_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:  # no Content-Length (chunked upload): stop as soon as it goes over
                    raise HTTPException(413, _too_large(limit))
            return message

        await self.app(scope, counted_receive, send)


def _too_large(limit: int) -> str:
    return f"Request too large (limit {limit // 1_000_000} MB)." if limit >= 1_000_000 else "Request too large."


async def _reply_413(send, limit: int) -> None:
    body = json.dumps({"detail": _too_large(limit)}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": 413, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RateLimiter:
    """At most `per_client` requests per visitor and `total` requests overall in any `window` seconds."""

    MAX_TRACKED = 10_000  # visitors kept in memory; beyond that, idle ones are forgotten

    def __init__(self, per_client: int, total: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.per_client, self.total, self.window, self.clock = per_client, total, window, clock
        self.hits: dict[str, deque[float]] = {}
        self.all_hits: deque[float] = deque()

    def check(self, client: str) -> float:
        """0 if the request may go ahead (and counts it), else the seconds to wait."""
        now = self.clock()
        _expire(self.all_hits, now - self.window)
        mine = self.hits.setdefault(client, deque())
        _expire(mine, now - self.window)
        for hits, cap in ((mine, self.per_client), (self.all_hits, self.total)):
            if len(hits) >= cap:
                return max(hits[0] + self.window - now, 1.0)
        mine.append(now)
        self.all_hits.append(now)
        if len(self.hits) > self.MAX_TRACKED:
            self.hits = {k: v for k, v in self.hits.items() if v and v[-1] > now - self.window}
        return 0.0


def _expire(hits: deque[float], before: float) -> None:
    while hits and hits[0] <= before:
        hits.popleft()


def client_key(request: Request) -> str:
    """Best guess at the visitor's address behind Render's proxy (see the module docstring for why it can be faked)."""
    h = request.headers
    ip = h.get("cf-connecting-ip") or h.get("true-client-ip") or h.get("x-forwarded-for", "").split(",")[0]
    ip = ip.strip() or (request.client.host if request.client else "unknown")
    return ip[:64]


def rate_limited(name: str):
    """FastAPI dependency: 429 with Retry-After when app.state.limits[name] says no."""

    def dependency(request: Request) -> None:
        wait = request.app.state.limits[name].check(client_key(request))
        if wait:
            raise HTTPException(
                429,
                "Too many requests. Please wait a minute and try again.",
                headers={"Retry-After": str(int(wait) + 1)},
            )

    return dependency

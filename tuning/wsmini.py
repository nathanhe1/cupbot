"""Minimal dependency-free asyncio websocket client (RFC 6455, client side).

Enough for market-data feeds: text/binary frames, fragmentation, ping/pong,
close. Python 3.9+, stdlib only, so it runs on the Mac without installs.
Honours HTTPS_PROXY via CONNECT when set.
"""
import asyncio
import base64
import os
import ssl
import struct
import time
from urllib.parse import urlparse

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC11B53"


class WSClosed(Exception):
    pass


class WS:
    def __init__(self, reader, writer):
        self.r = reader
        self.w = writer
        self.closed = False

    async def send(self, data):
        if isinstance(data, str):
            payload, op = data.encode(), 0x1
        else:
            payload, op = data, 0x2
        await self._send_frame(op, payload)

    async def _send_frame(self, op, payload):
        mask = os.urandom(4)
        n = len(payload)
        hdr = bytes([0x80 | op])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack("!H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack("!Q", n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        self.w.write(hdr + mask + masked)
        await self.w.drain()

    async def recv(self):
        """Return (text_or_bytes, recv_monotonic_ns). recv time = when the
        first byte of the final frame was read, the fairest latency stamp."""
        buf = b""
        first_op = None
        while True:
            h = await self.r.readexactly(2)
            t_ns = time.time_ns()
            fin = h[0] & 0x80
            op = h[0] & 0x0F
            n = h[1] & 0x7F
            if n == 126:
                n = struct.unpack("!H", await self.r.readexactly(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", await self.r.readexactly(8))[0]
            if h[1] & 0x80:  # server should not mask, but handle it
                mk = await self.r.readexactly(4)
                data = bytes(b ^ mk[i & 3] for i, b in enumerate(await self.r.readexactly(n)))
            else:
                data = await self.r.readexactly(n) if n else b""
            if op == 0x9:  # ping
                await self._send_frame(0xA, data)
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                self.closed = True
                raise WSClosed(data[2:].decode(errors="replace") if len(data) > 2 else "")
            if op in (0x1, 0x2):
                first_op = op
            buf += data
            if fin:
                return (buf.decode() if first_op == 0x1 else buf), t_ns

    async def close(self):
        try:
            await self._send_frame(0x8, struct.pack("!H", 1000))
        except Exception:
            pass
        self.w.close()


async def connect(url, headers=None, timeout=10.0):
    u = urlparse(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "wss" else 80)
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None) if u.scheme == "wss" else None
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy and u.scheme == "wss":
        import socket
        p = urlparse(proxy)
        raw = socket.create_connection((p.hostname, p.port), timeout=timeout)
        raw.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = raw.recv(4096)
            if not chunk:
                break
            resp += chunk
        if b" 200" not in resp.split(b"\r\n", 1)[0]:
            raw.close()
            raise ConnectionError(f"proxy CONNECT failed: {resp[:120]!r}")
        raw.setblocking(False)
        r, w = await asyncio.wait_for(asyncio.open_connection(sock=raw, ssl=ctx, server_hostname=host), timeout)
    else:
        r, w = await asyncio.wait_for(asyncio.open_connection(host, port, ssl=ctx, server_hostname=host if ctx else None), timeout)
    sock = w.get_extra_info("socket")
    if sock is not None:
        import socket
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = base64.b64encode(os.urandom(16)).decode()
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    req = [f"GET {path} HTTP/1.1", f"Host: {host}", "Upgrade: websocket", "Connection: Upgrade",
           f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13", "User-Agent: cupbot-tuning/1"]
    for k, v in (headers or {}).items():
        req.append(f"{k}: {v}")
    w.write(("\r\n".join(req) + "\r\n\r\n").encode())
    await w.drain()
    status = await asyncio.wait_for(r.readline(), timeout)
    if b" 101" not in status:
        rest = await r.read(2000)
        raise ConnectionError(f"websocket upgrade failed: {status!r} {rest[:300]!r}")
    while (await r.readline()) not in (b"\r\n", b""):
        pass
    return WS(r, w)

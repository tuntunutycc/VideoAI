#!/usr/bin/env python3
"""Small TCP front for the Streamlit UI.

Remote QA reaches this host over a NetBird relay. Streamlit's default
socket buffers let one slow peer queue several megabytes, which inflates
latency until the browser gives up. This proxy caps the send buffer on
the client side and forwards bytes unchanged, including WebSockets.
"""

import asyncio
import os
import socket

LISTEN_PORT = int(os.environ.get("EDGE_LISTEN", "8501"))
UPSTREAM = (
    os.environ.get("EDGE_UPSTREAM_HOST", "127.0.0.1"),
    int(os.environ.get("EDGE_UPSTREAM_PORT", "8502")),
)
# Kernel doubles the requested value for bookkeeping. 32 KiB requested
# is about 64 KiB of in-flight data per connection.
SNDBUF = int(os.environ.get("EDGE_SNDBUF", str(32 * 1024)))


def tune_client_socket(sock: socket.socket) -> None:
    """Lock a small send buffer before any bytes are queued.

    Set this on the listening socket as well as the accepted socket.
    TCP only inherits the limit when it is set before listen/accept.
    """
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SNDBUF)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NOTSENT_LOWAT, 16 * 1024)
    except OSError:
        pass


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(16 * 1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass


async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    sock = client_writer.get_extra_info("socket")
    if isinstance(sock, socket.socket):
        tune_client_socket(sock)
    try:
        remote_reader, remote_writer = await asyncio.open_connection(*UPSTREAM)
    except OSError as exc:
        peer = client_writer.get_extra_info("peername")
        print(f"upstream {UPSTREAM[0]}:{UPSTREAM[1]} unavailable for {peer}: {exc}", flush=True)
        client_writer.close()
        return

    remote_sock = remote_writer.get_extra_info("socket")
    if isinstance(remote_sock, socket.socket):
        try:
            remote_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    tasks = {
        asyncio.create_task(pipe(client_reader, remote_writer)),
        asyncio.create_task(pipe(remote_reader, client_writer)),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for writer in (client_writer, remote_writer):
        try:
            writer.close()
        except OSError:
            pass


def listen_socket(family: int, address: tuple) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    tune_client_socket(sock)
    sock.bind(address)
    sock.listen(128)
    sock.setblocking(False)
    return sock


async def main() -> None:
    loop = asyncio.get_running_loop()
    servers = []
    binds = (
        (socket.AF_INET, ("0.0.0.0", LISTEN_PORT)),
        (socket.AF_INET6, ("::", LISTEN_PORT)),
    )
    for family, address in binds:
        try:
            sock = listen_socket(family, address)
        except OSError as exc:
            print(f"not listening on {address}: {exc}", flush=True)
            continue
        servers.append(await asyncio.start_server(handle, sock=sock))
        print(f"edge listening on {address[0]}:{address[1]} -> {UPSTREAM[0]}:{UPSTREAM[1]}", flush=True)
    if not servers:
        raise SystemExit("no listen sockets")
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())

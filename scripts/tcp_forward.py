"""Small TCP forwarder for local debugging.

Example:
    python3 scripts/tcp_forward.py --listen-host 0.0.0.0 --listen-port 9000 --target-host 172.20.224.1 --target-port 9000
"""

from __future__ import annotations

import argparse
import select
import socket
import threading
from typing import Iterable


def pipe_bidirectional(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        while True:
            readable, _, exceptional = select.select(sockets, [], sockets)
            if exceptional:
                break
            for src in readable:
                dst = right if src is left else left
                data = src.recv(1024 * 64)
                if not data:
                    return
                dst.sendall(data)
    finally:
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


def handle_client(client: socket.socket, addr: tuple[str, int], target_host: str, target_port: int) -> None:
    try:
        target = socket.create_connection((target_host, int(target_port)), timeout=10)
    except OSError as exc:
        print(f"connect target failed addr={addr} target={target_host}:{target_port} error={exc}", flush=True)
        client.close()
        return

    print(f"forward connected addr={addr} -> {target_host}:{target_port}", flush=True)
    pipe_bidirectional(client, target)
    print(f"forward disconnected addr={addr}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward one TCP port to another host:port.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=9000)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, default=9000)
    parser.add_argument("--backlog", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.listen_host, int(args.listen_port)))
        server.listen(max(1, int(args.backlog)))
        print(
            f"forward listening {args.listen_host}:{args.listen_port} -> {args.target_host}:{args.target_port}",
            flush=True,
        )
        while True:
            client, addr = server.accept()
            threading.Thread(
                target=handle_client,
                args=(client, addr, args.target_host, int(args.target_port)),
                daemon=True,
            ).start()


if __name__ == "__main__":
    raise SystemExit(main())

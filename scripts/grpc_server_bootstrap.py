"""Minimal public bootstrap for the protected MaixSense service."""

import multiprocessing

from scripts.grpc_server import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

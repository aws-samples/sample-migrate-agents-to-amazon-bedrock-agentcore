# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import threading


def interruptible_wait(seconds: float) -> None:
    """Wait interruptibly; time.sleep cannot be interrupted in minute-long polls."""
    threading.Event().wait(seconds)

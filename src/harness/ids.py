import os
import threading
import time
import uuid

_lock = threading.Lock()
_last_ms = 0
_counter = 0


def uuid7() -> str:
    """Time-ordered UUID (version 7), monotonic within a process (RFC 9562, counter in rand_a).

    Rows sort by creation time (ERD modelling rule 6), including ids created in the same millisecond.
    """
    global _last_ms, _counter
    with _lock:
        ms = int(time.time() * 1000)
        if ms <= _last_ms:                 # same millisecond or clock stepped back: keep order
            ms = _last_ms
            _counter += 1
            if _counter > 0xFFF:           # counter exhausted: move to the next millisecond
                ms, _counter = ms + 1, 0
        else:
            _counter = 0
        _last_ms = ms
        counter = _counter
    b = bytearray(ms.to_bytes(6, "big") + counter.to_bytes(2, "big") + os.urandom(8))
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))

"""Bounded thread-safe login throttling for the local HTTP service."""
from collections import deque
from threading import Lock
from time import monotonic

class LoginGuard:
    def __init__(self, attempts=5, window=60, capacity=2048, clock=monotonic):
        if attempts < 1 or window <= 0 or capacity < 1:
            raise ValueError('Некорректные параметры ограничения входа')
        self.attempts = attempts
        self.window = window
        self.capacity = capacity
        self.clock = clock
        self.entries = {}
        self.lock = Lock()

    def reserve(self, address):
        now = self.clock()
        with self.lock:
            for key in list(self.entries):
                queue = self.entries[key]
                while queue and now - queue[0] >= self.window:
                    queue.popleft()
                if not queue:
                    del self.entries[key]
            if address not in self.entries:
                if len(self.entries) >= self.capacity:
                    return False
                self.entries[address] = deque()
            queue = self.entries[address]
            if len(queue) >= self.attempts:
                return False
            queue.append(now)
            return True

    def clear(self, address):
        with self.lock:
            self.entries.pop(address, None)

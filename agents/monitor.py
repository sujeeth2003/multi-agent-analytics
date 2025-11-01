"""Thread-safe monitoring of agent nodes: call counts, failures, latency percentiles."""
import threading
from collections import defaultdict


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.lat = defaultdict(list)
        self.fail = defaultdict(int)
        self.errors = []

    def record(self, node, seconds, ok=True, error=None):
        with self.lock:
            self.lat[node].append(seconds)
            if not ok:
                self.fail[node] += 1; self.errors.append((node, error))


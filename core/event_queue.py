"""
EventQueue: chronological priority queue with tie-breaking by event type priority.
FillEvents always process before BarEvents at the same timestamp.
"""

import heapq
from typing import List, Tuple, Any
from datetime import datetime
from core.events import Event, EventType, EVENT_PRIORITY


class EventQueue:
    def __init__(self):
        self._heap: List[Tuple] = []
        self._counter = 0  # tie-breaker for equal (timestamp, priority) pairs

    def put(self, event: Event) -> None:
        priority = EVENT_PRIORITY[event.event_type]
        ts = event.timestamp.timestamp()  # float seconds since epoch
        heapq.heappush(self._heap, (ts, priority, self._counter, event))
        self._counter += 1

    def get(self) -> Event:
        if self.empty():
            raise IndexError("EventQueue is empty")
        _, _, _, event = heapq.heappop(self._heap)
        return event

    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def peek_timestamp(self) -> datetime | None:
        if self.empty():
            return None
        ts_float = self._heap[0][0]
        return datetime.utcfromtimestamp(ts_float)

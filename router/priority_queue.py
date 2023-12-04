import queue
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(order=True)
class QueueItem:
    node_id: int = field(compare=False)
    heuristic: timedelta = field(compare=False)
    timestamp: datetime = field(compare=False)
    priority: datetime = field(init=False)

    def __post_init__(self):
        self.priority = self.timestamp + self.heuristic


class PriorityQueue:
    def __init__(self) -> None:
        self.size = 0
        self._queue = queue.PriorityQueue()

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        item = self._get()
        return item

    def put(self, item: QueueItem):
        self._queue.put(item)
        self.size += 1

    def _get(self) -> QueueItem:
        self.size -= 1
        return self._queue.get()

import enum

from dataclasses import dataclass

class DynamicEventType(enum.Enum):
    TELEPORT = "teleport"
    SMOOTH_MOVE = "smooth_move"


@dataclass(frozen=True)
class DynamicEvent:
    step: int
    type: DynamicEventType
    duration: int = 10


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    events: tuple[DynamicEvent, ...]
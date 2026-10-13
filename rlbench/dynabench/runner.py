from rlbench.backend.scene import Scene
from rlbench.dynabench.scenario import (
    DynamicEvent,
    DynamicEventType,
    DynamicScenario,
)


class ScenarioRunner:
    def __init__(self, scene: Scene, scenario: DynamicScenario):
        self.scene = scene
        self.scenario = scenario

        self.current_step = 0
        self.started_events = set()
        self.active_events = {}
        self.completed_events = set()

    def reset(self):
        self.current_step = 0
        self.started_events.clear()
        self.active_events.clear()
        self.completed_events.clear()

    def step(self):
        """Advance the scenario by one simulation step."""

        for event_id, event in enumerate(self.scenario.events):
            if event_id in self.started_events:
                continue

            if event.step == self.current_step:
                self._start_event(event_id, event)

        for event_id, event in list(self.active_events.items()):
            if self._advance_event(event):
                self.active_events.pop(event_id)
                self.completed_events.add(event_id)

        self.current_step += 1

    def _start_event(self, event_id: int, event: DynamicEvent):
        self.started_events.add(event_id)

        if event.type == DynamicEventType.TELEPORT:
            self.scene.teleport()
            self.completed_events.add(event_id)

        elif event.type == DynamicEventType.SMOOTH_MOVE:
            self.active_events[event_id] = event

        else:
            raise ValueError(
                f"Unsupported dynamics event type: {event.type}"
            )

    def _advance_event(self, event: DynamicEvent) -> bool:
        """Advance a running event; return True when it finishes."""

        if event.type == DynamicEventType.SMOOTH_MOVE:
            return self.scene.move_task_smoothly(
                total_steps=event.duration
            )

        raise ValueError(
            f"Cannot advance event type: {event.type}"
        )

    @property
    def is_finished(self) -> bool:
        """Whether all scenario events have completed."""
        return len(self.completed_events) == len(self.scenario.events)
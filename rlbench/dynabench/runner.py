from typing import Dict, Set, Tuple

from rlbench.backend.scene import Scene
from rlbench.dynabench.scenario import (
    DynamicEvent,
    DynamicEventType,
    DynamicScenario,
)


class ScenarioRunner:
    """Executes a dynamic scenario over cached scene layouts."""

    def __init__(self, scene: Scene, scenario: DynamicScenario):
        self.scene = scene
        self.scenario = scenario

        self.current_step = 0
        self.current_layout_index = 0

        self.started_events: Set[int] = set()
        self.completed_events: Set[int] = set()

        # event_id -> (event, target_layout_index)
        self.active_events: Dict[
            int, Tuple[DynamicEvent, int]
        ] = {}

        self._validate_layouts()

    def _validate_layouts(self) -> None:
        """Ensure that every event has a corresponding target layout."""
        required_layouts = len(self.scenario.events) + 1
        available_layouts = len(self.scene._layouts)

        if available_layouts < required_layouts:
            raise ValueError(
                f"Scenario '{self.scenario.name}' requires "
                f"{required_layouts} layouts, but the scene has "
                f"{available_layouts}. Pass num_layouts="
                f"{required_layouts} to Scene.init_episode()."
            )

    def reset(self) -> None:
        """Reset the runner to the initial layout and timestep."""
        self.current_step = 0
        self.current_layout_index = 0

        self.started_events.clear()
        self.active_events.clear()
        self.completed_events.clear()

        self.scene.reset_layout_transition()

    def step(self) -> None:
        """Advance the scenario by one simulation step."""
        for event_id, (event, target_index) in list(
            self.active_events.items()
        ):
            if self._advance_event(event, target_index):
                self.active_events.pop(event_id)
                self.completed_events.add(event_id)
                self.current_layout_index = target_index

        for event_id, event in enumerate(self.scenario.events):
            if event_id in self.started_events:
                continue

            if event.step == self.current_step:
                self._start_event(event_id, event)

        for event_id, (event, target_index) in list(
            self.active_events.items()
        ):
            if event_id in self.completed_events:
                continue

            if event.step != self.current_step:
                continue

            if self._advance_event(event, target_index):
                self.active_events.pop(event_id)
                self.completed_events.add(event_id)
                self.current_layout_index = target_index

        self.current_step += 1

    def _start_event(
        self,
        event_id: int,
        event: DynamicEvent,
    ) -> None:
        """Start an event using the next cached layout."""
        self.started_events.add(event_id)

        target_layout_index = self.current_layout_index + 1

        if target_layout_index >= len(self.scene._layouts):
            raise RuntimeError(
                f"Event {event_id} ('{event.type.value}') requires "
                f"layout {target_layout_index}, but only "
                f"{len(self.scene._layouts)} layouts are available."
            )

        if event.type == DynamicEventType.TELEPORT:
            self.scene.teleport(layout_index=target_layout_index)
            self.current_layout_index = target_layout_index
            self.completed_events.add(event_id)

        elif event.type == DynamicEventType.SMOOTH_MOVE:
            self.active_events[event_id] = (
                event,
                target_layout_index,
            )

        else:
            raise ValueError(
                f"Unsupported dynamics event type: {event.type}"
            )

    def _advance_event(
        self,
        event: DynamicEvent,
        target_layout_index: int,
    ) -> bool:
        """Advance a running event; return True when it finishes."""
        if event.type == DynamicEventType.SMOOTH_MOVE:
            return self.scene.move_task_smoothly(
                source_layout_index=self.current_layout_index,
                target_layout_index=target_layout_index,
                total_steps=event.duration,
            )

        raise ValueError(
            f"Cannot advance event type: {event.type}"
        )

    @property
    def is_finished(self) -> bool:
        """Whether all scenario events have completed."""
        return len(self.completed_events) == len(self.scenario.events)
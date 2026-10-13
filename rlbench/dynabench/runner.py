from typing import Dict, Set, Tuple

import numpy as np

from rlbench.backend.scene import Scene
from rlbench.dynabench.scenario import (
    DynamicEvent,
    DynamicEventType,
    DynamicScenario,
    ArmDisturbanceEvent,
)
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    ArmActionMode,
)


def assert_end_effector_action_mode(action_mode) -> None:
    """Reject action modes that don't accept end-effector pose commands."""

    arm_action_mode = getattr(
        action_mode,
        "arm_action_mode",
        None,
    )

    if not isinstance(
        arm_action_mode,
        (
            EndEffectorPoseViaIK,
            EndEffectorPoseViaPlanning,
        ),
    ):
        raise ValueError(
            "ARM_DISTURBANCE requires an end-effector pose action mode "
            "(EndEffectorPoseViaIK or EndEffectorPoseViaPlanning). "
            f"Received: {type(arm_action_mode).__name__}."
        )

class ScenarioRunner:
    """Executes a dynamic scenario over cached scene layouts."""

    def __init__(self, scene: Scene, scenario: DynamicScenario, action_mode: ArmActionMode) -> None:
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

        if any(
            event.type == DynamicEventType.ARM_DISTURBANCE
            for event in scenario.events
        ):
            assert_end_effector_action_mode(
                action_mode
            )

        self.active_disturbances: dict[int, ArmDisturbanceEvent] = {}

    def _validate_layouts(self) -> None:
        """Ensure that every event has a corresponding target layout."""
        required_layouts = 1 + self.scenario.n_scene_events
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

    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance the scenario by one agent step and modify its action."""

        # Start events scheduled for this agent step.
        for event_id, event in enumerate(self.scenario.events):
            if event_id in self.started_events:
                continue

            if event.step == self.current_step:
                self._start_event(event_id, event)

        for event_id, (event, target_index) in list(
            self.active_events.items()
        ):
            if self._advance_event(event, target_index):
                self.active_events.pop(event_id)
                self.completed_events.add(event_id)
                self.current_layout_index = target_index


        for event_id, event in self.active_disturbances.items():
            self._apply_arm_disturbance(action, event)

            if self.current_step >= event.step + event.duration:
                self.completed_events.add(event_id)

        self.current_step += 1
        return action

    def _start_event(self, event_id: int, event: DynamicEvent) -> None:
        self.started_events.add(event_id)

        if event.type == DynamicEventType.TELEPORT:
            target_index = self.current_layout_index + 1
            self.scene.teleport(layout_index=target_index)
            self.current_layout_index = target_index
            self.completed_events.add(event_id)

        elif event.type == DynamicEventType.SMOOTH_MOVE:
            target_index = self.current_layout_index + 1
            self.active_events[event_id] = (event, target_index)

        elif isinstance(event, ArmDisturbanceEvent):
            self._validate_arm_disturbance(event)
            self.active_disturbances[event_id] = event

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

    def _validate_arm_disturbance(self, event: DynamicEvent) -> None:
        if event.duration <= 0:
            raise ValueError(
                "Arm disturbance duration must be greater than zero."
            )

        offset = np.asarray(event.offset, dtype=float)
        if offset.shape != (7,):
            raise ValueError(
                f"Arm disturbance offset must have shape (7,), "
                f"but received {offset.shape}."
            )

        if not np.all(np.isfinite(offset)):
            raise ValueError("Arm disturbance offset must be finite.")

        if self.scene.robot.is_bimanual:
            if event.arm_idx not in (0, 1):
                raise ValueError(
                    "Bimanual arm disturbances require arm_idx=0 or arm_idx=1."
                )


    def _apply_arm_disturbance(
            self,
            action: np.ndarray,
            event: DynamicEvent,
    ) -> None:
        """Modify one arm's end-effector pose in place."""

        if action.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional action array, got {action.shape}."
            )

        offset = np.asarray(event.offset, dtype=float)

        # Single-arm action layout:
        # [x, y, z, qx, qy, qz, qw, gripper]
        #
        # Bimanual action layout:
        # [arm0 pose (7), arm0 gripper, arm1 pose (7), arm1 gripper]
        if self.scene.robot.is_bimanual:
            expected_size = 16

            if action.size != expected_size:
                raise ValueError(
                    "Expected a 16-element bimanual action containing two "
                    "8-element arm blocks, but received "
                    f"{action.size} elements."
                )

            start = event.arm_idx * 8
        else:
            expected_size = 8

            if action.size != expected_size:
                raise ValueError(
                    "Expected an 8-element single-arm action "
                    "(7D pose + gripper), but received "
                    f"{action.size} elements."
                )

            start = 0

        # Preserve the original disturbance schedule:
        # ramp linearly from zero to the full offset, then hold it.
        elapsed = self.current_step - event.step

        if elapsed < 0:
            return

        alpha = min(elapsed / event.duration, 1.0)

        action[start:start + 7] += alpha * offset
import enum

from dataclasses import dataclass

class DynamicEventType(enum.Enum):
    TELEPORT = "teleport"
    SMOOTH_MOVE = "smooth_move"
    ARM_DISTURBANCE = "arm_disturbance"

@dataclass(frozen=True)
class DynamicEvent:
    """Base class for scenario events."""
    step: int


@dataclass(frozen=True)
class SceneDynamicEvent(DynamicEvent):
    """An event that changes the scene layout."""
    type: DynamicEventType
    duration: int = 10

    def __post_init__(self):
        if self.type not in (
            DynamicEventType.TELEPORT,
            DynamicEventType.SMOOTH_MOVE,
        ):
            raise ValueError(
                f"Invalid scene event type: {self.type}"
            )

        if self.step < 0:
            raise ValueError("Event step must be non-negative.")

        if self.duration <= 0:
            raise ValueError("Event duration must be positive.")


@dataclass(frozen=True)
class ArmDisturbanceEvent(DynamicEvent):
    """An event that offsets an arm's end-effector action."""
    duration: int = 50
    arm_idx: int | None = None
    offset: tuple[float, ...] = (
        0.04, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0,
    )

    @property
    def type(self) -> DynamicEventType:
        return DynamicEventType.ARM_DISTURBANCE

    def __post_init__(self):
        if self.step < 0:
            raise ValueError("Event step must be non-negative.")

        if self.duration <= 0:
            raise ValueError("Disturbance duration must be positive.")

        if len(self.offset) != 7:
            raise ValueError(
                "Arm disturbance offset must contain 7 values."
            )

@dataclass(frozen=True)
class DynamicScenario:
    name: str
    description: str
    events: tuple[DynamicEvent, ...]

    def __post_init__(self):
        sorted_events = sorted(self.events, key=lambda e: e.step)

        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            following = sorted_events[i + 1]

            # Only scene transitions occupy a transition interval.
            if isinstance(current, SceneDynamicEvent):
                duration = (
                    1
                    if current.type == DynamicEventType.TELEPORT
                    else current.duration
                )

                if current.step + duration > following.step:
                    raise ValueError(
                        f"Dynamic events overlap: "
                        f"{current} and {following}"
                    )

    @property
    def n_scene_events(self) -> int:
        return sum(
            isinstance(event, SceneDynamicEvent)
            for event in self.events
        )


teleport_at_t5 = SceneDynamicEvent(step=5, type=DynamicEventType.TELEPORT)
teleport_at_t50 = SceneDynamicEvent(step=50, type=DynamicEventType.TELEPORT)
teleport_at_t100 = SceneDynamicEvent(step=100, type=DynamicEventType.TELEPORT)

interpolate_at_t5 = SceneDynamicEvent(step=5, type=DynamicEventType.SMOOTH_MOVE, duration=10)
interpolate_at_t50 = SceneDynamicEvent(step=50, type=DynamicEventType.SMOOTH_MOVE, duration=10)
interpolate_at_t100 = SceneDynamicEvent(step=100, type=DynamicEventType.SMOOTH_MOVE, duration=10)

single_teleport_scenario_t5 = DynamicScenario(
    name="single_teleport_scenario_early",
    description="A single teleport event at step 5. Appropriate for most tasks. Dynamics happens early in the episode, ie. with long buffer before first object interaction.",
    events=(teleport_at_t5,),
)
single_teleport_scenario_t50 = DynamicScenario(
    name="single_teleport_scenario_mid",
    description="A single teleport event at step 50. Appropriate for most tasks. Dynamics happens mid-episode, ie. with little buffer before first object interaction.",
    events=(teleport_at_t50,),
)
single_teleport_scenario_t100 = DynamicScenario(
    name="single_teleport_scenario_late",
    description="A single teleport event at step 100. Appropriate for multi-object tasks like PlaceCups. Dynamics happen shortly after first object interaction.",
    events=(teleport_at_t100,),
)

single_smooth_move_scenario_t5 = DynamicScenario(
    name="single_smooth_move_scenario_early",
    description="A single smooth move event at step 5. Appropriate for most tasks. Dynamics happens early in the episode, ie. with long buffer before first object interaction.",
    events=(interpolate_at_t5,),
)

single_smooth_move_scenario_t50 = DynamicScenario(
    name="single_smooth_move_scenario_mid",
    description="A single smooth move event at step 50. Appropriate for most tasks. Dynamics happens mid-episode, ie. with little buffer before first object interaction.",
    events=(interpolate_at_t50,),
)

single_smooth_move_scenario_t100 = DynamicScenario(
    name="single_smooth_move_scenario_late",
    description="A single smooth move event at step 100. Appropriate for multi-object tasks like PlaceCups. Dynamics happen shortly after first object interaction.",
    events=(interpolate_at_t100,),
)

double_teleport_scenario_single_skill = DynamicScenario(
    name="double_teleport_scenario_single_skill",
    description="Two teleport events at steps 5 and 50. Appropriate for most tasks.",
    events=(teleport_at_t5, teleport_at_t50),
) 

double_teleport_scenario_two_skill = DynamicScenario(
    name="double_teleport_scenario",
    description="Two teleport events at steps 5 and 100. Appropriate for multi-object tasks like PlaceCups.",
    events=(teleport_at_t5, teleport_at_t100),
)

double_smooth_move_scenario_single_skill = DynamicScenario(
    name="double_smooth_move_scenario_single_skill",
    description="Two smooth move events at steps 5 and 50. Appropriate for most tasks.",
    events=(interpolate_at_t5, interpolate_at_t50),
)

double_smooth_move_scenario_two_skill = DynamicScenario(
    name="double_smooth_move_scenario",
    description="Two smooth move events at steps 5 and 100. Appropriate for multi-object tasks like PlaceCups.",
    events=(interpolate_at_t5, interpolate_at_t100),
)

disturb_right_arm = ArmDisturbanceEvent(
    step=100,
    duration=50,
    arm_idx=0,
    offset=(0.04, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0),
)

disturb_left_arm = ArmDisturbanceEvent(
    step=100,
    duration=50,
    arm_idx=1,
    offset=(0.04, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0),
)

right_arm_disturbance_scenario = DynamicScenario(
    name="right_arm_disturbance_t100",
    description="Gradually offset the right end-effector action from step 100.",
    events=(disturb_right_arm,),
)

left_arm_disturbance_scenario = DynamicScenario(
    name="left_arm_disturbance_t100",
    description="Gradually offset the left end-effector action from step 100.",
    events=(disturb_left_arm,),
)




SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        single_teleport_scenario_t5,
        single_teleport_scenario_t50,
        single_teleport_scenario_t100,
        single_smooth_move_scenario_t5,
        single_smooth_move_scenario_t50,
        single_smooth_move_scenario_t100,
        double_teleport_scenario_single_skill,
        double_teleport_scenario_two_skill,
        double_smooth_move_scenario_single_skill,
        double_smooth_move_scenario_two_skill,
        right_arm_disturbance_scenario,
        left_arm_disturbance_scenario,
    )
}

def load_scenario(name: str) -> DynamicScenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"Unknown scenario '{name}'. Available scenarios: {available}"
        ) from None
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
    description: str
    events: tuple[DynamicEvent, ...]

    def __post_init__(self):
        for event in self.events:
            if event.step < 0:
                raise ValueError(
                    f"Event step must be nonnegative: {event}"
                )

            if event.type == DynamicEventType.SMOOTH_MOVE and event.duration <= 0:
                raise ValueError(
                    f"Smooth movement duration must be positive: {event}"
                )

        sorted_events = sorted(self.events, key=lambda e: e.step)

        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]

            duration = (
                1
                if current.type == DynamicEventType.TELEPORT
                else current.duration
            )

            if current.step + duration > next_event.step:
                raise ValueError(
                    f"Dynamic events overlap: {current} and {next_event}"
                )


teleport_at_t5 = DynamicEvent(step=5, type=DynamicEventType.TELEPORT)
teleport_at_t50 = DynamicEvent(step=50, type=DynamicEventType.TELEPORT)
teleport_at_t100 = DynamicEvent(step=100, type=DynamicEventType.TELEPORT)

interpolate_at_t5 = DynamicEvent(step=5, type=DynamicEventType.SMOOTH_MOVE, duration=10)
interpolate_at_t50 = DynamicEvent(step=50, type=DynamicEventType.SMOOTH_MOVE, duration=10)
interpolate_at_t100 = DynamicEvent(step=100, type=DynamicEventType.SMOOTH_MOVE, duration=10)

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
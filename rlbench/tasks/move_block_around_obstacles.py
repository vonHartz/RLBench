from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import NothingGrasped, DetectedCondition
from rlbench.backend.custom_conditions import BottlesUprightCondition


MAX_ROTATION = 30  # Grad


class MoveBlockAroundObstacles(Task):
    def init_task(self) -> None:
        self.block = Shape('move_block')
        self.wine_bottle_obstacles = [
            Shape('wine_bottle%d' % i)
            for i in range(6)
        ]

        # Drop-off zone
        self.drop_off_zone = Shape('drop_zone')
        self.drop_sensor = ProximitySensor('drop_zone_sensor')

        self.register_graspable_objects([self.block])

        self.register_success_conditions([
            DetectedCondition(
                self.block,
                self.drop_sensor
            ),
            NothingGrasped(self.robot.gripper),
            BottlesUprightCondition(self.wine_bottle_obstacles),
        ])

    def init_episode(self, index: int) -> List[str]:
        return [
            'move the block around the obstacles',
            'transport the cube through the obstacle course',
            'pick up the cube and place it in the target zone'
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        max_rot = np.deg2rad(MAX_ROTATION)

        return [0, 0, -max_rot], [0, 0, max_rot]

    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self.block, self.drop_off_zone]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)

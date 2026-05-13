from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped


class StackWine(Task):

    def init_task(self):
        self.wine_bottle = Shape('wine_bottle')
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.wine_bottle])
        self.register_success_conditions(
            [DetectedCondition(self.wine_bottle, self.success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        return ['stack wine bottle',
                'slide the bottle onto the wine rack',
                'put the wine away',
                'leave the wine on the shelf',
                'grasp the bottle and put it away',
                'place the wine bottle on the wine rack']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]
        
    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self.wine_bottle, self.success_sensor]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)


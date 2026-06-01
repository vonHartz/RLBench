from typing import List

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from collections import defaultdict

class BimanualStraightenRope(BimanualTask):

    def init_task(self) -> None:
        self.head = Shape('head')
        self.tail = Shape('tail')
        self.head_sensor = ProximitySensor('success_head')
        self.tail_sensor = ProximitySensor('success_tail')
        self.register_success_conditions(
            [DetectedCondition(self.head, self.head_sensor),
             DetectedCondition(self.tail, self.tail_sensor)])

        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(3):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['straighten rope',
                'pull the rope straight',
                'grasping each end of the rope in turn, leave the rope straight'
                ' on the table',
                'pull each end of the rope until is is straight',
                'tighten the rope',
                'pull the rope tight']

    def variation_count(self) -> int:
        return 1
    
    def get_low_dim_state(self) -> np.ndarray:
        head_pose = self.head.get_pose()
        tail_pose = self.tail.get_pose()
        head_sensor_pose = self.head_sensor.get_pose()
        tail_sensor_pose = self.tail_sensor.get_pose() 
        return np.concatenate([head_pose, tail_pose, head_sensor_pose, tail_sensor_pose])

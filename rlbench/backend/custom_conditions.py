import numpy as np
from scipy.spatial.transform import Rotation
from rlbench.backend.conditions import Condition


class BottlesUprightCondition(Condition):
    def __init__(self, bottles, max_tilt_angle_deg=30):
        self.bottles = bottles
        self.max_tilt_angle = np.deg2rad(max_tilt_angle_deg)

    def condition_met(self):
        world_up = np.array([0, 0, 1])

        for bottle in self.bottles:

            pose = bottle.get_pose()
            quat = pose[3:]

            rot = Rotation.from_quat(quat)

            # mirror bottle rotation since bottle local coordinate system is placed upside-down
            bottle_up = -rot.apply([0, 0, 1])

            angle = np.arccos(
                np.clip(
                    np.dot(
                        bottle_up,
                        world_up
                    ),
                    -1.0,
                    1.0
                )
            )

            if angle > self.max_tilt_angle:
                return False, False
        return True, False
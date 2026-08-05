import numpy as np
from scipy.spatial.transform import Rotation
from rlbench.backend.conditions import Condition


class BottlesUprightCondition(Condition):
    def __init__(self, bottles, max_translation_distance=0.03, max_tilt_angle_deg=30):
        self.bottles = bottles
        self.max_tilt_angle = np.deg2rad(max_tilt_angle_deg)
        self.max_translation_distance = max_translation_distance

        # Initiale Positionen speichern
        self.initial_positions = None

    def condition_met(self):
        # Init at first iteration
        if self.initial_positions is None:
            self.initial_positions = [
                np.array(bottle.get_position())
                for bottle in self.bottles
            ]
        world_up = np.array([0, 0, 1])

        for bottle, initial_pos in zip(self.bottles, self.initial_positions):
            # Check translation compared to initial pose
            current_pos = np.array(bottle.get_position())
            distance = np.linalg.norm(current_pos - initial_pos)
            if distance > self.max_translation_distance:
                print("Distance Exceeded:", distance)
                return True, False

            # Check orientation
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
                print("Angle", angle)
                return True, False

        return False, False
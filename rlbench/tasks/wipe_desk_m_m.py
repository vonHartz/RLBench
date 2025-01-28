from typing import List
import numpy as np
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import EmptyCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

DIRT_POINTS = 50


class WipeDeskMM(Task):

    def init_task(self) -> None:
        self.dirt_spots = []
        self.sponge = Shape('sponge')
        self.sensor = ProximitySensor('sponge_sensor')
        self.register_graspable_objects([self.sponge])

        boundaries = [Shape('dirt_boundary')]
        _, _, self.z_boundary = boundaries[0].get_position()
        self.b = SpawnBoundary(boundaries)

    def init_episode(self, index: int) -> List[str]:
        mode = np.random.randint(0, 2)
        if True or mode == 1:
            wp2 = Dummy('waypoint2')
            wp4 = Dummy('waypoint4')
            pose2 = wp2.get_position()
            pose4 = wp4.get_position()
            wp2.set_position(pose4)
            wp4.set_position(pose2)
            wp3 = CartesianPath('waypoint3')

            reversed_path = CartesianPath.create(path_color=[0, 0, 1])
            reversed_path.set_name('waypoint3overwrite')
            reversed_path.set_parent(wp3.get_parent())
            print("parents ", wp3.get_parent()._handle, reversed_path.get_parent()._handle)
            print("parents ", wp3.get_parent().get_pose(), reversed_path.get_parent().get_pose())

            parent = wp3.get_parent()
            # reversed_path.set_pose(wp3.get_pose(relative_to=parent), relative_to=parent)

            reversed_path.set_orientation(wp3.get_orientation())

            print("================================")
            po_2 = list(wp2.get_position()) + list(wp2.get_orientation())
            po_3 = list(wp3.get_position()) + list(wp3.get_orientation())
            po_4 = list(wp4.get_position()) + list(wp4.get_orientation())
            sp = list(self.sponge.get_position()) + list(self.sponge.get_orientation())
            wp1 = Dummy('waypoint1')
            po_1 = list(wp1.get_position()) + list(wp1.get_orientation())
            rp = list(reversed_path.get_position()) + list(reversed_path.get_orientation())
            print("sponge ", sp)
            print("wp1 ", po_1)  # Set control point orient to this?
            print("wp2 ", po_2)
            print("wp3 ", po_3)
            print("reversed_path ", rp)
            print("wp4 ", po_4)

            num_samples = 50
            sampled_poses = []
            for i in range(num_samples + 1):
                rel_dist = i / num_samples
                pos, ori = wp3.get_pose_on_path(rel_dist)
                # ori[0] = ori[0] + 2 * np.pi
                ori = po_1[3:]
                sampled_poses.append(pos + ori)

            reversed_poses = sampled_poses[::-1]

            print(sampled_poses)
            print(reversed_poses)


            # wp3.cut_control_points(0, -1)
            # wp3.insert_control_points(sampled_poses)

            reversed_path.insert_control_points(sampled_poses)
            # reversed_path.insert_control_points(reversed_poses)
            # reversed_path.insert_control_points(
            #     [po_2, po_4]
            # )

            p3 = wp3.get_pose()
            # reversed_path.set_pose(p3)
            # wp3.remove()
            # wp3.set_name('oldwaypoint3')
            self.reversed_path = reversed_path
            # wp3.remove()
            # wp3._handle, reversed_path._handle = reversed_path._handle, wp3._handle

            
        self._place_dirt()
        self.register_success_conditions([EmptyCondition(self.dirt_spots)])
        return ['wipe dirt off the desk',
                'use the sponge to clean up the desk',
                'remove the dirt from the desk',
                'grip the sponge and wipe it back and forth over any dirt you '
                'see',
                'clean up the mess',
                'wipe the dirt up']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        for d in self.dirt_spots:
            if self.sensor.is_detected(d):
                self.dirt_spots.remove(d)
                d.remove()

    def cleanup(self) -> None:
        for d in self.dirt_spots:
            d.remove()
        self.dirt_spots = []
        self.reversed_path.remove()

    def _place_dirt(self):
        for i in range(DIRT_POINTS):
            spot = Shape.create(type=PrimitiveShape.CUBOID,
                                size=[.005, .005, .001],
                                mass=0, static=True, respondable=False,
                                renderable=True,
                                color=[0.58, 0.29, 0.0])
            spot.set_parent(self.get_base())
            spot.set_position([-1, -1, self.z_boundary + 0.001])
            self.b.sample(spot, min_distance=0.00,
                          min_rotation=(0.00, 0.00, 0.00),
                          max_rotation=(0.00, 0.00, 0.00))
            self.dirt_spots.append(spot)
        self.b.clear()

    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self.sponge]  # + self.dirt_spots
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)
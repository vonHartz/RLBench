import importlib
import pickle

from itertools import product            
import os
from os import listdir
from os.path import join, exists
from typing import List

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig

import logging


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str, bimanual=False):
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    try:
        if bimanual:
            mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
        else:
            mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled."
            % name) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)) from e
    return task_class


def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0) -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    
    if amount > len(examples[from_episode_number:]):
        raise RuntimeError('You specified from_episode_number=%d, but only %d examples were available', from_episode_number,  len(examples))

    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        # language description
        episode_descriptions_f = join(example_path, VARIATION_DESCRIPTIONS)
        if exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        else:
            descriptions = ["unknown task description"]

        num_steps = len(obs)

        camera_names = obs_config.camera_configs.keys()

        data_types = ["rgb", "depth", "mask"]
        full_camera_names = map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types))
        
        for camera_name, _ in full_camera_names:
            data_path = os.path.join(example_path, camera_name)
            if num_steps != len(os.listdir(data_path)):
                print(f"not sufficent data points {data_path} expected {num_steps} was {len(os.listdir(data_path))}")
                #raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            # descriptions
            obs[i].misc['descriptions'] = descriptions
           
            for camera_name, camera_config in obs_config.camera_configs.items():                

                if camera_config.rgb:
                    data_path = os.path.join(example_path, f"{camera_name}_rgb")
                    image_name = f"rgb_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    image = np.array(_resize_if_needed(Image.open(image_path), camera_config.image_size))
                    obs[i].perception_data[f"{camera_name}_rgb"] = image
                
                if camera_config.depth or camera_config.point_cloud:
                    data_path = os.path.join(example_path, f"{camera_name}_depth")
                    image_name = f"depth_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    image = image_to_float_array( _resize_if_needed(Image.open(image_path), camera_config.image_size),DEPTH_SCALE)

                    if camera_config.depth:
                        if camera_config.depth_in_meters:
                            near = obs[i].misc[f'{camera_name}_camera_near']
                            far = obs[i].misc[f'{camera_name}_camera_far']
                            depth_image_m = near + image * (far - near)
                            obs[i].perception_data[f"{camera_name}_depth"] = camera_config.depth_noise.apply(depth_image_m)
                        else:                        
                            obs[i].perception_data[f"{camera_name}_depth"] = camera_config.depth_noise.apply(image)

                    if camera_config.point_cloud:
                        near = obs[i].misc[f'{camera_name}_camera_near']
                        far = obs[i].misc[f'{camera_name}_camera_far']
                        depth_image_m = near + image * (far - near)

                        obs[i].perception_data[f"{camera_name}_point_cloud"] = VisionSensor.pointcloud_from_depth_and_camera_params(
                        depth_image_m,
                        obs[i].misc[f'{camera_name}_camera_extrinsics'],
                        obs[i].misc[f'{camera_name}_camera_intrinsics'])
               

                if camera_config.mask:
                    data_path = os.path.join(example_path, f"{camera_name}_mask")
                    image_name = f"mask_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    obs[i].perception_data[f"{camera_name}_mask"] = rgb_handles_to_mask(np.array(_resize_if_needed(Image.open(image_path), camera_config.image_size)))
          

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None



        demos.append(obs)
    return demos


def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0, q1 : array-like, shape (4,)
        Quaternions in [x, y, z, w] format.
    t : float
        Interpolation ratio in [0, 1].

    Returns
    -------
    q : ndarray, shape (4,)
        Interpolated quaternion.
    """
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = np.dot(q0, q1)

    # Take shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return normalize_quaternion(q)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


def interpolate_pose(pose0, pose1, t):
    """
    Interpolate between two poses.

    Parameters
    ----------
    pose0, pose1 : ndarray, shape (7,)
        Pose format: [x, y, z, qx, qy, qz, qw]
    t : float
        Interpolation ratio in [0, 1].

    Returns
    -------
    pose : ndarray, shape (7,)
        Interpolated pose.
    """
    pose0 = np.asarray(pose0, dtype=np.float64)
    pose1 = np.asarray(pose1, dtype=np.float64)

    # Linear interpolation for position
    p0, p1 = pose0[:3], pose1[:3]
    p = (1.0 - t) * p0 + t * p1

    # SLERP for orientation
    q0, q1 = pose0[3:], pose1[3:]
    q = slerp(q0, q1, t)

    return np.concatenate([p, q])
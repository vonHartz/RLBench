from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class DynamicObjectState:
    handle: int
    obj: object
    relative_pose: np.ndarray
    world_pose: np.ndarray


@dataclass
class DynamicConfiguration:
    root_pose: np.ndarray
    objects: Dict[int, DynamicObjectState]
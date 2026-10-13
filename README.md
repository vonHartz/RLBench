# DynaBench: Dynamic Evaluation Protocol for RLBench

DynaBench is built on [RLBench](https://github.com/stepjam/RLBench).
It also includes the bimanual tasks from [RLBench2](https://github.com/markusgrotz/rlbench/), as well as bug fixes from [PerAct](https://github.com/MohitShridhar/RLBench/tree/peract).
You can find bibtex entries for all papers at the bottom of this document.

Please find the readme for RLBench below for details on installation, headless rendering, etc.

## Scene Dynamics
DynaBench implements two types of scene dynamics: teleportation and linear interpolation.
Teleportation samples a new (valid) task instance and teleports all scene objects into the new configuration.
Linear interpolation also samples a new instance but then moves the object to their new poses over a predefined number of time steps.
Multiple such dynamic transitions can be chained.

By interpolating two arbitrary valid task instances, DynaBench randomizes both speed and movement direction of the task objects.
Movement speed is controlled by the number of time steps (default=10), but interacts with the dispersion of the object poses across task instances.
For example, for PlaceCups, the scene objects move with speeds between 0 m/s and 1.4 m/s. (By default, RLBench is controlled at 20 Hz.)

## Arm Disturbances
We also include the option to disturb an arm during task execution by adding an offset to the predicted end-effector pose that ramps up over a specified time horizon and then stays at that offset level.

Both scene dynamics and arm disturbance are easiest to evaluate using our predefined scenarios.

## Scenarios
We predefine dynamic scenarios in `rlbench/dynabench/scenario.py`.
If set on a `task_env`, a dynamics scenario is automatically executed in sync with calls to `task_env.step` using a ScenarioRunner from `rlbench/dynabench/runner.py`.

You can load and use them like this:
```python
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import OpenMicrowave
from rlbench.task_environment import TaskEnvironment

action_mode = MoveArmThenGripper(
  arm_action_mode=EndEffectorPoseViaIK(
    absolute_mode=False,
    frame="world",
    ),
  gripper_action_mode=Discrete(),
)
env = Environment(action_mode)
env.launch()
task_env = env.get_task(OpenMicrowave)

# Smooth scene dynamics scenario
task_env.set_scenario("single_smooth_move_scenario_early")

# Arm disturbance scenario
# task_env.set_scenario("right_arm_disturbance_t100")

_, obs = task_env.reset()

policy = ...

step_no = 0
while True:
  action = policy.predict(obs)
  next_obs, reward, done = task_env.step(action)

  if done:
    break

  obs = next_obs
```

These are the currently available scenarios.

| Scenario name | Dynamic type | Event steps | Number of transitions | Duration per transition (steps) | Total transition steps | Layouts required | Intended use |
|---|---|---:|---:|---:|---:|---:|---|
| `single_teleport_scenario_early` | Teleport | 5 | 1 | Instantaneous | 0 | 2 | Most tasks; early change with a long buffer before the first object interaction. |
| `single_teleport_scenario_mid` | Teleport | 50 | 1 | Instantaneous | 0 | 2 | Most tasks; mid-episode change with little buffer before the first object interaction. |
| `single_teleport_scenario_late` | Teleport | 100 | 1 | Instantaneous | 0 | 2 | Multi-object tasks such as PlaceCups; change shortly after the first object interaction. |
| `single_smooth_move_scenario_early` | Smooth move | 5 | 1 | 10 | 10 | 2 | Most tasks; early, gradual scene change with a long buffer before the first object interaction. |
| `single_smooth_move_scenario_mid` | Smooth move | 50 | 1 | 10 | 10 | 2 | Most tasks; mid-episode gradual change with little buffer before the first object interaction. |
| `single_smooth_move_scenario_late` | Smooth move | 100 | 1 | 10 | 10 | 2 | Multi-object tasks such as PlaceCups; gradual change shortly after the first object interaction. |
| `double_teleport_scenario_single_skill` | Teleport | 5, 50 | 2 | Instantaneous | 0 | 3 | Most tasks; two discrete scene changes within a relatively short episode. |
| `double_teleport_scenario` | Teleport | 5, 100 | 2 | Instantaneous | 0 | 3 | Multi-object tasks such as PlaceCups; two discrete scene changes spaced farther apart. |
| `double_smooth_move_scenario_single_skill` | Smooth move | 5, 50 | 2 | 10 | 20 | 3 | Most tasks; two gradual scene changes within a relatively short episode. |
| `double_smooth_move_scenario` | Smooth move | 5, 100 | 2 | 10 | 20 | 3 | Multi-object tasks such as PlaceCups; two gradual scene changes spaced farther apart. |
| `right_arm_disturbance_t100` | Arm disturbance | 100 | 0 scene transitions; 1 arm disturbance | 50-step ramp | 50-step ramp | 1 | Gradually offsets the right end-effector action beginning at step 100. |
| `left_arm_disturbance_t100` | Arm disturbance | 100 | 0 scene transitions; 1 arm disturbance | 50-step ramp | 50-step ramp | 1 | Gradually offsets the left end-effector action beginning at step 100. |

### Manual Example
You can also manually apply scene dynamics from your own evaluation script by accessing the `task_env._scene` object. For example:

```python
from dataclasses import dataclass
from loguru import logger

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import OpenMicrowave
from rlbench.task_environment import TaskEnvironment


@dataclass
class Dynamics:
    teleport_at_t: int | None = None
    smooth_move_at_t: int | None = None
    smooth_move_duration: int = 10


def apply_scene_dynamics(task_env: TaskEnvironment, step_no: int, dynamics: Dynamics):
    if (
        dynamics.smooth_move_at_t is not None
        and step_no >= dynamics.smooth_move_at_t + dynamics.smooth_move_duration
    ):
        pass
    elif dynamics.smooth_move_at_t is not None and step_no > dynamics.smooth_move_at_t:
        logger.info(
            "Applying smooth move dynamics for {} steps.",
            dynamics.smooth_move_duration,
        )
        task_env._scene.move_task_smoothly(
            total_steps=dynamics.smooth_move_duration
        )

    if dynamics.teleport_at_t is not None and step_no == dynamics.teleport_at_t:
        logger.info("Applying teleportation dynamics.")
        task_env._scene.teleport()

action_mode = MoveArmThenGripper(
  arm_action_mode=EndEffectorPoseViaIK(
    absolute_mode=False,
    frame="world",
    ),
  gripper_action_mode=Discrete(),
)
env = Environment(action_mode)
env.launch()
task_env = env.get_task(OpenMicrowave)
_, obs = task_env.reset()

policy = ...

dynamics = Dynamics(teleport_at_t=5)

step_no = 0
while True:
  action = policy.predict(obs)
  apply_scene_dynamics(task_env, step_no, dynamics)
  next_obs, reward, done = task_env.step(action)

  if done:
    break

  obs = next_obs
```

## Bimanual Tasks
For bimanual manipulation, import a bimanual task (e.g. `from rlbench.bimanual_tasks import BimanualHandoverItem`) and pass `robot_setup=dual_panda` upon environment initialization.

Environment dynamics and arm disturbance can be applied identical to the unimanual tasks.


# RLBench: Robot Learning Benchmark [![Unit Tests](https://github.com/stepjam/RLBench/workflows/Unit%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Task Tests](https://github.com/stepjam/RLBench/workflows/Task%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Discord](https://img.shields.io/discord/694945190867370155.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/DXPCjmd)

![task grid image missing](readme_files/task_grid.png)

## RLBench

**RLBench** is an ambitious large-scale benchmark and learning environment 
designed to facilitate research in a number of vision-guided manipulation
research areas, including: reinforcement learning, imitation learning,
multi-task learning, geometric computer vision, and in particular, 
few-shot learning. [Click here for website and paper.](https://sites.google.com/corp/view/rlbench)

**Contents:**
- [Install](#install)
- [Running Headless](#running-headless)
- [Getting Started](#getting-started)
    - [Few-Shot Learning and Meta Learning](#few-shot-learning-and-meta-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Sim-to-Real](#sim-to-real)
    - [Imitation Learning](#imitation-learning)
    - [Multi-Task Learning](#multi-task-learning)
    - [RLBench Gym](#rlbench-gym)
    - [Swapping Arms](#swapping-arms)
- [Tasks](#tasks)
- [Task Building](#task-building)
- [Gotchas!](#gotchas)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Install

RLBench is built around PyRep and V-REP. First head to the 
[PyRep github](https://github.com/stepjam/PyRep) page and install.

**If you previously had PyRep installed, you will need to update your installation!**

Hopefully you have now installed PyRep and have run one of the PyRep examples.
Now lets install RLBench:

```bash
pip install -r requirements.txt
pip install -e .
```
Make sure to install using the editable flag (-e)!

## Running Headless

If you are running on a machine without display (i.e. Cloud VMs, compute clusters),
you can refer to the following guide to run RLBench headlessly with rendering.

### Initial setup

First, configure your X config. This should only be done once to set up.

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf
```

Leave out `--use-display-device=None` if the GPU is headless, i.e. if it has no display outputs.

### Running X

Then, whenever you want to run RLBench, spin up X.

```bash
# nohup and disown is important for the X server to keep running in the background
sudo nohup X :99 & disown
```

Test if your display works using glxgears.

```bash
DISPLAY=:99 glxgears
```

If you have multiple GPUs, you can select your GPU by doing the following.

```bash
DISPLAY=:99.<gpu_id> glxgears
```

### Running X without sudo

To spin up X with non-sudo users, edit file '/etc/X11/Xwrapper.config' and replace line:

```
allowed_users=console
```
with lines:
```
allowed_users=anybody
needs_root_rights=yes
```
If the file does not exist already, you can create it.

## Getting Started

The benchmark places particular emphasis on few-shot learning and meta learning 
due to breadth of tasks available, though it can be used in numerous ways. Before using RLBench, 
checkout the [Gotchas](#gotchas) section.

### Few-Shot Learning and Meta Learning

We have created splits of tasks called 'Task Sets', which consist of a 
collection of X training tasks and 5 tests tasks. Here X can be 10, 25, 50, or 95.
For example, to work on the task set with 10 training tasks, we import `FS10_V1`:

```python
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import FS10_V1

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode)
env.launch()

train_tasks = FS10_V1['train']
test_tasks = FS10_V1['test']
task_to_train = np.random.choice(train_tasks, 1)[0]
task = env.get_task(task_to_train)
task.sample_variation()  # random variation
descriptions, obs = task.reset()
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
```

A full example can be seen in [examples/few_shot_rl.py](examples/few_shot_rl.py).

### Reinforcement Learning

```python
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
```

A full example can be seen in [examples/single_task_rl.py](examples/single_task_rl.py).
If you would like to bootstrap from demonstrations, then take a look at [examples/single_task_rl_with_demos.py](examples/single_task_rl_with_demos.py).


### Sim-to-Real

```python
import numpy as np
from rlbench import Environment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import OpenDoor

# We will borrow some from the tests dir
rand_config = VisualRandomizationConfig(
    image_directory='../tests/unit/assets/textures')

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(
    action_mode, randomize_every=RandomizeEvery.EPISODE, 
    frequency=1, visual_randomization_config=rand_config)

env.launch()

task = env.get_task(OpenDoor)
descriptions, obs = task.reset()
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
```

A full example can be seen in [examples/single_task_rl_domain_randomization.py](examples/single_task_rl_domain_randomization.py).

### Imitation Learning

```python
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget

# To use 'saved' demos, set the path below
DATASET = 'PATH/TO/YOUR/DATASET'

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode, DATASET)
env.launch()

task = env.get_task(ReachTarget)

demos = task.get_demos(2)  # -> List[List[Observation]]
demos = np.array(demos).flatten()

batch = np.random.choice(demos, replace=False)
batch_images = [obs.left_shoulder_rgb for obs in batch]
predicted_actions = predict_action(batch_images)
ground_truth_actions = [obs.joint_velocities for obs in batch]
loss = behaviour_cloning_loss(ground_truth_actions, predicted_actions)

```

A full example can be seen in [examples/imitation_learning.py](examples/imitation_learning.py).

### Multi-Task Learning

We have created splits of tasks called 'Task Sets', which consist of a 
collection of X training tasks. Here X can be 15, 30, 55, or 100.
For example, to work on the task set with 15 training tasks, we import `MT15_V1`:

```python
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import MT15_V1

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode)
env.launch()

train_tasks = MT15_V1['train']
task_to_train = np.random.choice(train_tasks, 1)[0]
task = env.get_task(task_to_train)
task.sample_variation()  # random variation
descriptions, obs = task.reset()
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
```

A full example can be seen in [examples/multi_task_learning.py](examples/multi_task_learning.py).

### RLBench Gym

RLBench is __Gym__ compatible! Ensure you have gym installed (`pip3 install gym`).

Simply select your task of interest from [rlbench/tasks/](rlbench/tasks/), and
then load the task by using the task name (e.g. 'reach_target') followed by
the observation mode: 'state' or 'vision'.

```python
import gym
import rlbench.gym

env = gym.make('reach_target-state-v0')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()
```

A full example can be seen in [examples/rlbench_gym.py](examples/rlbench_gym.py).

### Swapping Arms

The default Franka Panda Arm _can_ be swapped out for another. This can be
useful for those who have custom tasks or want to perform sim-to-real 
experiments on the tasks. However, if you swap out the arm, then we can't 
guarantee that the task will be solvable.
For example, the Mico arm has a very small workspace in comparison to the
Franka.

**For benchmarking, the arm should remain as the Franka Panda.**

Currently supported arms:

- Franka Panda arm with Franka gripper `(franka)`
- Mico arm with Mico gripper `(mico)`
- Jaco arm with 3-finger Jaco gripper `(jaco)`
- Sawyer arm with Baxter gripper `(sawyer)`
- UR5 arm with Robotiq 85 gripper `(ur5)`

You can then swap out the arm using `robot_configuration`:

```python
env = Environment(action_mode=action_mode, robot_setup='sawyer')
```

A full example (using the Sawyer) can be seen in [examples/swap_arm.py](examples/swap_arm.py).

_Don't see the arm that you want to use?_ Your first step is to make sure it is
in PyRep, and if not, then you can follow the instructions for importing new
arm on the PyRep GitHub page. After that, feel free to open an issue and 
we can being it in to RLBench for you.

## Tasks

To see a full list of all tasks, [see here](rlbench/tasks).

To see gifs of each of the tasks, [see here](https://drive.google.com/drive/folders/1TqbulbbCEqVBd6SBHatphFlUK2JQLkYu?usp=sharing).

## Task Building

The task building tool is the interface for users who wish to create new tasks 
to be added to the RLBench task repository. Each task has 2 associated files: 
a V-REP model file (_.ttm_), which holds all of the scene information and demo 
waypoints, and a python (_.py_) file, which is responsible for wiring the 
scene objects to the RLBench backend, applying variations, defining success
criteria, and adding other more complex task behaviours.

Video tutorial series [here](https://www.youtube.com/watch?v=bKaK_9O3v7Y&list=PLsffAlO5lBTRiBwnkw2-x0U7t6TrNCkfc)!

In-depth text tutorials:
- [Simple Task](tutorials/simple_task.md)
- [Complex Task](tutorials/complex_task.md)

## Gotchas!

- **Using low-dimensional task observations (rather than images):** RLBench was designed to be challenging, putting emphasis on vision rather than 
toy-based low dimensional inputs. Although each task does supply a low-dimensional
output this should be used with extreme caution!
    - Why? Imagine you are training a reinforcement learning agent to pick up a block; halfway through
    training, the block slips from the gripper and falls of the table. These low-dimensional values
    will now be out of distribution. I.e. RLBench does not safeguard against objects going out of the 
    workspace. This issue does not arise when using image-based observations. 
    
- **Using non-standard image size:** RLBench by default uses image observation sizes of 128x128.
When using an alternative size, be aware that you may need to collect your saved demonstrations again.
    - Why? If we instead specify a 64x64 image observation size to the `ObservationConfig` then the
    scene cameras will now render to that size. However, the saved demos on disk will now be **resized**
    to be 64x64.
    This resizing will of course mean that small artifacts may be present in stored demos
    that may not be present in the 'live' observations from the scene. Instead, prefer to re-collect demos
    using the image observation sized you plan to use in the 'live' environment.
    

## Contributing

New tasks using our task building tool, in addition to bug fixes, are very 
welcome! When building your task, please ensure that you run the task validator
in the task building tool.

A full contribution guide is coming soon!

## Acknowledgements

Models were supplied from turbosquid.com, cgtrader.com, free3d.com, 
thingiverse.com, and cadnav.com.

## Citation

```
@article{james2019rlbench,
  title={RLBench: The Robot Learning Benchmark \& Learning Environment},
  author={James, Stephen and Ma, Zicong and Rovick Arrojo, David and Davison, Andrew J.},
  journal={IEEE Robotics and Automation Letters},
  year={2020}
}

@inproceedings{grotz2024peract2,
  title={Peract2: Benchmarking and learning for robotic bimanual manipulation tasks},
  author={Grotz, Markus and Shridhar, Mohit and Chao, Yu-Wei and Asfour, Tamim and Fox, Dieter},
  booktitle={CoRL 2024 Workshop on Whole-body Control and Bimanual Manipulation: Applications in Humanoids and Beyond},
  year={2024}
}

@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation}, 
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}

@inproceedings{anonymous2026,
  title     = {One Hand Watches The Other: Dynamic Multi-Agent Bimanual Control for Sample-Efficient Manipulation in Dynamic Environments}, 
  author    = {Anonymous Authors},
}

```

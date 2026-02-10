# Tracking changes that might have broken the leIsaac

Very important to note that it requires lerobot = ["lerobot==0.4.2"]

## isaaclab2lerobot

scripts/convert/isaaclab2lerobot.py


```python
"""
This script converts IsaacLab HDF5 datasets into LeRobot Dataset v2 format.
Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
Please install the following specific versions of the dependencies:
pip install lerobot==0.3.3
pip install numpy==1.26.0
"""

import argparse
import os
from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v2.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
        " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="EverNorif/so101_test_orange_pick",
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default=None,
    help="Task description. If not provided, will use the description defined in the task.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# default arguments
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.env_utils import get_task_type
from leisaac.utils.robot_utils import build_feature_from_env


def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has less than 10 frames, skip it")
        return False
    episode_list = split_episode(episode, num_frames)
    # skip the first 5 frames
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        predefined_task = frame.pop("task")
        dataset.add_frame(frame=frame, task=predefined_task if task is None else task)
    return True

def convert_isaaclab_to_lerobot():
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
    )
    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)
    
    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    now_episode_index = 0
    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")
        for episode_name in tqdm(episode_names, desc="Processing each episode"):
            episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
            if not episode.success:
                print(f"Episode {episode_name} is not successful, skip it")
                continue
            valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    if args_cli.push_to_hub:
        dataset.push_to_hub()

    print("Finished converting IsaacLab dataset to LeRobot dataset")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
```


### OLD isaaclab2lerobot

scripts/convert/isaaclab2lerobot.py

Copy stuff from the old leisaac

```python
import os

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

"""
NOTE: Please use the environment of lerobot.

Because lerobot is rapidly developing, we don't guarantee the compatibility for the latest version of lerobot.
Currently, the commit we used is https://github.com/huggingface/lerobot/tree/v0.3.3
"""

# Feature definition for single-arm so101_follower
SINGLE_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# Feature definition for bi-arm so101_follower
BI_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# preprocess actions and joint pos
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    joint_pos = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min
    return joint_pos


def process_single_arm_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group["actions"])
        joint_pos = np.array(demo_group["obs/joint_pos"])
        front_images = np.array(demo_group["obs/front"])
        wrist_images = np.array(demo_group["obs/wrist"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess actions and joint pos
    actions = preprocess_joint_pos(actions)
    joint_pos = preprocess_joint_pos(joint_pos)

    assert actions.shape[0] == joint_pos.shape[0] == front_images.shape[0] == wrist_images.shape[0]
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.front": front_images[frame_index],
            "observation.images.wrist": wrist_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def process_bi_arm_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group["actions"])
        left_joint_pos = np.array(demo_group["obs/left_joint_pos"])
        right_joint_pos = np.array(demo_group["obs/right_joint_pos"])
        left_images = np.array(demo_group["obs/left_wrist"])
        right_images = np.array(demo_group["obs/right_wrist"])
        top_images = np.array(demo_group["obs/top"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess actions and joint pos
    actions = preprocess_joint_pos(actions)
    left_joint_pos = preprocess_joint_pos(left_joint_pos)
    right_joint_pos = preprocess_joint_pos(right_joint_pos)

    assert (
        actions.shape[0]
        == left_joint_pos.shape[0]
        == right_joint_pos.shape[0]
        == left_images.shape[0]
        == right_images.shape[0]
        == top_images.shape[0]
    )
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": np.concatenate([left_joint_pos[frame_index], right_joint_pos[frame_index]]),
            "observation.images.left_wrist": left_images[frame_index],
            "observation.images.top": top_images[frame_index],
            "observation.images.right_wrist": right_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def convert_isaaclab_to_lerobot():
    """NOTE: Modify the following parameters to fit your own dataset"""
    repo_id = "EverNorif/so101_test_orange_pick"
    robot_type = "so101_follower"  # so101_follower, bi_so101_follower
    fps = 30
    hdf5_root = "./datasets"
    hdf5_files = [os.path.join(hdf5_root, "dataset.hdf5")]
    task = "Grab orange and place into plate"
    push_to_hub = False

    """parameters check"""
    assert robot_type in [
        "so101_follower",
        "bi_so101_follower",
    ], "robot_type must be so101_follower or bi_so101_follower"

    """convert to LeRobotDataset"""
    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=SINGLE_ARM_FEATURES if robot_type == "so101_follower" else BI_ARM_FEATURES,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} is not successful, skip it")
                    continue

                if robot_type == "so101_follower":
                    valid = process_single_arm_data(dataset, task, demo_group, demo_name)
                elif robot_type == "bi_so101_follower":
                    valid = process_bi_arm_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saving episode {now_episode_index} successfully")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()

```

## isaaclab2lerobotv3

scripts/convert/isaaclab2lerobotv3.py

```python
"""This script converts IsaacLab HDF5 datasets into LeRobot Dataset v3 format.

Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
Please install the following specific versions of the dependencies:

pip install lerobot==0.4.2
pip install numpy==1.26.0

"""

import argparse
import os
from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v3.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
        " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="EverNorif/so101_test_orange_pick",
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default=None,
    help="Task description. If not provided, will use the description defined in the task.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# default arguments
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.env_utils import get_task_type
from leisaac.utils.robot_utils import build_feature_from_env


def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has less than 10 frames, skip it")
        return False

    episode_list = split_episode(episode, num_frames)
    # skip the first 5 frames
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        if task is not None:
            frame["task"] = task
        dataset.add_frame(frame=frame)
    return True

def convert_isaaclab_to_lerobot():
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
    )
    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    now_episode_index = 0
    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")
        for episode_name in tqdm(episode_names, desc="Processing each episode"):
            episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
            if not episode.success:
                print(f"Episode {episode_name} is not successful, skip it")
                continue
            valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    dataset.finalize()

    if args_cli.push_to_hub:
        dataset.push_to_hub()

    print("Finished converting IsaacLab dataset to LeRobot dataset")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
```

## lerobot2isaaclab

This script did not exist before

scripts/convert/lerobot2isaaclab.py

```python
"""
Convert a local LeRobot dataset folder (v2.1 episode-based layout) to a single HDF5 file.
Extracts selected columns (default: 'action'), applies denormalization to 'action'.
"""

import argparse
import datetime as dt
import json
import os
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

ISAACLAB_LIMITS_DEG = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10.0, 100.0),
]
LEROBOT_LIMITS_DEG = [
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (0.0, 100.0),
]


def denormalize_lerobot_to_isaaclab_radians(joint_values_lerobot: np.ndarray) -> np.ndarray:
    """
    LeRobot normalized degrees -> IsaacLab joint limits (degrees) -> radians.
    """
    joint_values_lerobot = np.asarray(joint_values_lerobot, dtype=np.float32)

    if joint_values_lerobot.ndim != 2:
        raise ValueError(f"Expected 2D array (T,D), got {joint_values_lerobot.shape}")

    dimension = joint_values_lerobot.shape[1]
    if dimension < 6:
        raise ValueError(f"Expected D>=6, got D={dimension}")

    result_deg = joint_values_lerobot.copy()

    # Map first 6 joints (or 6+6 for bimanual)
    if dimension == 12:
        # bimanual
        for arm_offset in (0, 6):
            for joint_index in range(6):
                isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
                le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
                column_index = arm_offset + joint_index
                result_deg[:, column_index] = (result_deg[:, column_index] - le_min) / (le_max - le_min) * (
                    isa_max - isa_min
                ) + isa_min
    else:
        # single arm
        for joint_index in range(6):
            isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
            le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
            result_deg[:, joint_index] = (result_deg[:, joint_index] - le_min) / (le_max - le_min) * (
                isa_max - isa_min
            ) + isa_min

    return result_deg * (np.pi / 180.0)


def generate_stats_if_missing(dataset_root: Path):
    """
    Check if meta/episodes_stats.jsonl exists. If not, generate all stats.
    """
    episodes_stats_path = dataset_root / "meta/episodes_stats.jsonl"
    stats_path = dataset_root / "meta/stats.json"

    if episodes_stats_path.exists() and stats_path.exists():
        print(f"Stats files found in {dataset_root / 'meta'}, skipping generation.")
        return

    print("Missing stats files! Generating them now...")

    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_root}")

    with open(info_path) as info_file:
        info = json.load(info_file)
    features = info.get("features", {})

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    all_stats = []
    episode_stats_list = []

    for parquet_path in tqdm(parquet_files, desc="Computing stats"):
        dataframe = pd.read_parquet(parquet_path)
        episode_data = {}
        for column_name in dataframe.columns:
            if column_name not in features:
                continue
            column_values = dataframe[column_name].values
            # Handle object columns (lists/arrays)
            if (
                column_values.dtype == object
                and len(column_values) > 0
                and isinstance(column_values[0], (list, tuple, np.ndarray))
            ):
                with suppress(Exception):
                    column_values = np.stack(column_values)
            episode_data[column_name] = column_values

        try:
            stats = compute_episode_stats(episode_data, features)
            all_stats.append(stats)

            stats_with_index = stats.copy()
            stats_with_index["episode_index"] = int(parquet_path.stem.split("_")[-1])
            episode_stats_list.append(stats_with_index)
        except Exception as e:
            print(f"Error computing stats for {parquet_path}: {e}")
            raise e

    print("Aggregating statistics...")
    aggregated = aggregate_stats(all_stats)

    def numpy_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(stats_path, "w") as stats_file:
        json.dump(aggregated, stats_file, indent=4, default=numpy_converter)

    with open(episodes_stats_path, "w") as episodes_stats_file:
        for stat in episode_stats_list:
            clean_stat = json.loads(json.dumps(stat, default=numpy_converter))
            output_item = {
                "episode_index": clean_stat["episode_index"],
                "stats": {k: v for k, v in clean_stat.items() if k != "episode_index"},
            }
            episodes_stats_file.write(json.dumps(output_item) + "\n")

    print(f"Stats generated at {episodes_stats_path}")


def convert_lerobot_folder_to_hdf5(
    lerobot_dir: str,
    output_hdf5_path: str,
    column_keys: list[str] = ["action"],
    list_available_keys: bool = False,
    use_all_keys: bool = False,
):
    dataset_root = Path(lerobot_dir).expanduser().resolve()
    output_path = Path(output_hdf5_path).expanduser().resolve()

    # 1. Ensure stats exist
    generate_stats_if_missing(dataset_root)

    print(f"Loading LeRobotDataset from {dataset_root}...")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    if use_all_keys:
        if dataset.num_episodes > 0:
            first_idx = dataset.episode_data_index["from"][0].item()
            # Get keys from the first frame
            column_keys = list(dataset[first_idx].keys())
            print(f"[Info] Using all available keys: {column_keys}")
        else:
            print("[Warning] Dataset appears empty, cannot determine keys.")

    if list_available_keys:
        print("\n[Available Keys in Dataset]")
        if dataset.num_episodes > 0:
            # Peek at first frame
            first_idx = dataset.episode_data_index["from"][0].item()
            keys = list(dataset[first_idx].keys())
            for k in keys:
                print(f"  - {k}")
        else:
            print("  (Dataset is empty)")
        return

    # 2. Setup output HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as output_hdf5_file:
        output_hdf5_file.attrs["source_dir"] = str(dataset_root)
        output_hdf5_file.attrs["created_at"] = dt.datetime.now().isoformat()
        output_hdf5_file.attrs["convert_to_isaaclab_radians"] = True

        # Write meta files (optional but good for provenance)
        meta_group = output_hdf5_file.create_group("meta")
        if (dataset_root / "meta").exists():
            for meta_file_path in (dataset_root / "meta").glob("*"):
                if meta_file_path.is_file():
                    try:
                        content = meta_file_path.read_text(encoding="utf-8", errors="ignore")
                        meta_group.create_dataset(
                            meta_file_path.name, data=content, dtype=h5py.string_dtype(encoding="utf-8")
                        )
                    except Exception:
                        pass

        # 3. Iterate episodes and save selected columns
        for episode_index in tqdm(range(dataset.num_episodes), desc="Converting to HDF5"):
            # Load frames
            start_index = dataset.episode_data_index["from"][episode_index].item()
            end_index = dataset.episode_data_index["to"][episode_index].item()

            frames = [dataset[i] for i in range(start_index, end_index)]
            if not frames:
                continue

            # Create Group with new naming convention
            episode_group_name = f"data/demo_{episode_index}"
            episode_group = output_hdf5_file.require_group(episode_group_name)

            # Set frames count (based on first requested key found, or total frames)
            episode_group.attrs["num_frames"] = len(frames)

            for key in column_keys:
                # Extract specific column
                try:
                    values_list = [frame[key] for frame in frames]
                except KeyError:
                    print(f"Warning: Key '{key}' not found in episode {episode_index}, skipping.")
                    continue

                # Stack
                if isinstance(values_list[0], torch.Tensor):
                    values_array = torch.stack(values_list).numpy()
                else:
                    values_array = np.array(values_list)

                if key == "action":
                    values_array = denormalize_lerobot_to_isaaclab_radians(values_array)

                # Save compressed
                # Sanitize key for HDF5 dataset name
                safe_key = key.replace("/", "_")

                # Handle string data types (e.g. 'task')
                if values_array.dtype.kind in ("U", "S"):
                    episode_group.create_dataset(
                        safe_key,
                        data=values_array.astype("S"),  # Convert to fixed-length bytes (simplest for compatibility)
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    episode_group.create_dataset(
                        safe_key,
                        data=values_array,
                        compression="gzip",
                        compression_opts=4,
                    )

    print(f"[OK] Wrote HDF5 to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compact LeRobot to HDF5 converter")
    parser.add_argument("--lerobot_dir", type=str, required=True, help="Path to local LeRobot dataset")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to output HDF5")
    parser.add_argument(
        "--column_keys", type=str, nargs="+", default=["action"], help="Column names to extract (default: action)"
    )
    parser.add_argument("--all_keys", action="store_true", help="Extract all available keys (overrides --column_keys)")
    parser.add_argument("--list_keys", action="store_true", help="List available column keys in the dataset and exit")

    args = parser.parse_args()

    convert_lerobot_folder_to_hdf5(
        lerobot_dir=args.lerobot_dir,
        output_hdf5_path=args.output_hdf5,
        column_keys=args.column_keys,
        list_available_keys=args.list_keys,
        use_all_keys=args.all_keys,
    )


if __name__ == "__main__":
    main()
```

## teleop_se3_agent

scripts/environments/teleoperation/teleop_se3_agent.py

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(
                args_cli.dataset_file
            ), "the dataset file already exists, please use '--resume' to resume recording"
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager or lerobot recorder manager
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # create controller
    if args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader

        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader

        teleop_interface = BiSO101Leader(
            env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    teleop_interface.display_controls()
    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not interrupted:
            # run everything in inference mode
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                actions = teleop_interface.advance()
                if should_reset_task_success:
                    print("Task Success!!!")
                    should_reset_task_success = False
                    if args_cli.record:
                        manual_terminate(env, True)
                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    if start_record_state:
                        if args_cli.record:
                            print("Stop Recording!!!")
                        start_record_state = False
                    if args_cli.record:
                        manual_terminate(env, False)
                    # print out the current demo count if it has changed
                    if (
                        args_cli.record
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        > current_recorded_demo_count
                    ):
                        current_recorded_demo_count = (
                            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        )
                        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                    if (
                        args_cli.record
                        and args_cli.num_demos > 0
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        >= args_cli.num_demos
                    ):
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                        break

                elif actions is None:
                    env.render()
                # apply actions
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True
                    env.step(actions)
                if rate_limiter:
                    rate_limiter.sleep(env)
            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # finalize the recorder manager
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

```

## Dependencies

source/leisaac/pyproject.toml

```toml
[build-system]
requires = ["setuptools", "wheel", "toml"]
build-backend = "setuptools.build_meta"

[project]
name = "leisaac"
version = "0.3.0"
description = "leisaac: Lerobot & IsaacLab Integration"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "Yinghao Shuai", email = "yinghao.shuai@lightwheel.ai"}
]
maintainers = []
keywords = ["extension", "isaaclab", "leisaac", "lerobot"]
classifiers = [
    "Natural Language :: English",
    "Programming Language :: Python :: 3.11",
    "Isaac Sim :: 5.1.0",
]
requires-python = ">=3.10"
dependencies = [
    "deepdiff",
    "feetech-servo-sdk",
    "psutil",
    "pygame>=2.5.1,<2.7.0",
    "pyserial",
]

[project.urls]
Homepage = "https://github.com/LightwheelAI/leisaac"
Repository = "https://github.com/LightwheelAI/leisaac"

[tool.setuptools]
packages = {find = {}}
include-package-data = true
zip-safe = false


[project.optional-dependencies]
isaaclab = ["isaaclab[isaacsim,all]==2.3.0"]
gr00t = ["pyzmq>=27.0.0", "pydantic==2.10.6", "msgpack>=1.0.5"]
lerobot-async = ["grpcio==1.74.0", "protobuf==6.32.0"] # original isaaclab install protobuf version is 3.20.3
lerobot = ["lerobot==0.4.2"]
openpi = ["dm-tree>=0.1.8", "msgpack>=1.0.5", "numpy>=1.22.4,<2.0.0", "pillow>=9.0.0", "tree>=0.2.4", "websockets>=11.0"]
```
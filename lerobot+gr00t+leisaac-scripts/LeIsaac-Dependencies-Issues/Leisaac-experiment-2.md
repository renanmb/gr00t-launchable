# Experiment to make a New Dataset Handler

THe idea is to make LeRobot Optional and decoupled from the Dataset Handler. Since LeRobot can cause lots of issues and we need the Dict.

**The changes seem to have worked.**


Original Code for the Dataset Handler.

```python
import copy

from isaaclab.utils import configclass
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@configclass
class LeRobotDatasetCfg:
    """Configuration for the LeRobotDataset."""

    repo_id: str = None
    """Lerobot Dataset repository ID."""
    fps: int = 30
    """Lerobot Dataset frames per second."""
    robot_type: str = "so101_follower"
    """Robot type: so101_follower or bi_so101_follower, etc."""
    features: dict = None
    """Features for the LeRobotDataset."""
    action_align: bool = False
    """Whether the action shape equals to the joint number. If action align, we will convert action to lerobot limit range."""


class LeRobotDatasetHandler(DatasetFileHandlerBase):
    def __init__(self, cfg: LeRobotDatasetCfg):
        self._cfg = copy.deepcopy(cfg)
        self._lerobot_dataset = None
        self._demo_count = 0
        self._env_args = {}

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        if resume:
            self._lerobot_dataset = LeRobotDataset(
                repo_id=self._cfg.repo_id,
            )
        else:
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=self._cfg.repo_id,
                fps=self._cfg.fps,
                robot_type=self._cfg.robot_type,
                features=self._cfg.features,
            )
        self._env_args["env_name"] = env_name

    def open(self, file_path: str, mode: str = "r"):
        self._lerobot_dataset = LeRobotDataset(
            repo_id=self._cfg.repo_id,
        )

    def get_env_name(self) -> str | None:
        return self._env_args["env_name"]

    def add_frame(self, frame: dict):
        self._lerobot_dataset.add_frame(frame=frame)

    def flush(self):
        self._lerobot_dataset.save_episode(parallel_encoding=False)

    def clear(self):
        self._lerobot_dataset.clear_episode_buffer()

    def finalize(self):
        self._lerobot_dataset.finalize()

    def close(self):
        if self._lerobot_dataset is not None:
            self.finalize()
            self._lerobot_dataset = None

    # not used for now
    def write_episode(self, episode: EpisodeData):
        raise NotImplementedError("write_episode is not supported for LeRobotDatasetHandler")

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError("load_episode is not supported for LeRobotDatasetHandler")

    def get_num_episodes(self) -> int:
        raise NotImplementedError("get_num_episodes is not supported for LeRobotDatasetHandler")

```


## Changes to make LeRobot optional

Need to create a Generic Dataset Handler that mimics the LeRobotDatasetHandler.

**Step-1 --- Define a Generic Interface**

Since the **LeRobotDatasetHandler** already wraps the dataset, only need to change what happens inside **create** and **add_frame**. (Hopefully)

```python
# Create a simple fallback recorder if LeRobot is missing
class GenericDataRecorder:
    def __init__(self, repo_id, fps, features):
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.buffer = []

    def add_frame(self, frame):
        # Just store in memory or append to a list
        self.buffer.append(frame)

    def save_episode(self, **kwargs):
        # Save to a standard format like pickle or numpy
        import pickle
        with open(f"{self.repo_id}_episode.pkl", "wb") as f:
            pickle.dump(self.buffer, f)
        self.buffer = []

    def finalize(self):
        print("Dataset finalized.")
```

**Step-2 --- Update the Handler to be "Agnostic"**

Modify the **LeRobotDatasetHandler** to choose between the real **LeRobotDataset** and the generic one based on a flag or a failed import.

```python
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

class LeRobotDatasetHandler(DatasetFileHandlerBase):
    def __init__(self, cfg: LeRobotDatasetCfg):
        self._cfg = copy.deepcopy(cfg)
        self._dataset = None # Renamed from _lerobot_dataset

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        if HAS_LEROBOT:
            # Original LeRobot logic
            if resume:
                self._dataset = LeRobotDataset(repo_id=self._cfg.repo_id)
            else:
                self._dataset = LeRobotDataset.create(
                    repo_id=self._cfg.repo_id,
                    fps=self._cfg.fps,
                    robot_type=self._cfg.robot_type,
                    features=self._cfg.features,
                )
        else:
            # Decoupled logic: Use a simple dictionary logger
            print("LeRobot not found. Using GenericDataRecorder.")
            self._dataset = GenericDataRecorder(self._cfg.repo_id, self._cfg.fps, self._cfg.features)
```

**Step-3 --- Standardize the add_frame Call**

Make sure the methods are named in **GenericDataRecorder** the same as the ones in **LeRobotDataset** (add_frame, save_episode, check if there are more), the rest of your handler remains identical:

```python
def add_frame(self, frame: dict):
        # This works for BOTH LeRobot and GenericRecorder
        self._dataset.add_frame(frame=frame)

    def flush(self):
        # LeRobot uses save_episode; make sure your generic one does too
        self._dataset.save_episode(parallel_encoding=False)
```



**Step-4 --- Refactor the open Method**

If not using LeRobot, use **GenericDataRecorder** (or a similar loader).


```python
def open(self, file_path: str, mode: str = "r"):
    """Opens an existing dataset for reading or appending."""
    if HAS_LEROBOT:
        # Standard LeRobot initialization
        self._dataset = LeRobotDataset(
            repo_id=self._cfg.repo_id,
        )
    else:
        # Decoupled logic: Initialize your custom handler
        # You can use file_path here to load existing metadata
        self._dataset = GenericDataRecorder(
            repo_id=self._cfg.repo_id, 
            fps=self._cfg.fps, 
            features=self._cfg.features
        )
        if mode == "r":
            self._dataset.load_from_disk(file_path)
```


3. Why open is different from create

In the LeRobot library:

- ```.create()``` initializes a new repository/folder and writes the info.json (the metadata/features).

- ```__init__``` (called in open) expects the repository to already exist. It reads the existing metadata from the disk.

4. Handling the "Missing" Features

When you call **open** in the original code, the features dictionary is usually loaded from the files on disk rather than the config. To stay decoupled, the **GenericDataRecorder** needs to be able to do the same:

```python
class GenericDataRecorder:
    def __init__(self, repo_id, fps=None, features=None):
        self.repo_id = repo_id
        self.fps = fps
        self.features = features

    def load_from_disk(self, path):
        # Implementation to load your custom .npz or .json files
        # This ensures that even without LeRobot, your 'open' method 
        # populates the metadata correctly.
        pass
```

To fully decouple while keeping the logic clean:

**Rename** self._lerobot_dataset to self._dataset throughout the class.

**Use** a try-except block for the import.

**Wrap** the constructor in open and create with an if HAS_LEROBOT: check.
# Experiment to Remove the Dependency on LeRobot

LeRobot is used for very few things witthin leisaac and it can cause several issues.



Refactoring ```robot_utils.py``` to remove the dependency on LeRobot.

The File can be found at "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12

This is the culprit of the issue:

```python
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
```

Original code:

```python
def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, dataset_cfg: LeRobotDatasetCfg) -> dict:
    """
    Build the feature from the environment.
    """
    features = {}

    default_feature_joint_names = env.cfg.default_feature_joint_names
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    if action_dim != len(default_feature_joint_names):
        # [A bit tricky, currently works because the action dimension matches the joints only when we use leader control]
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
        dataset_cfg.action_align = False
    else:
        action_joint_names = default_feature_joint_names
        dataset_cfg.action_align = True
    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names)
    )

    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = dataset_cfg.fps
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features
```


Refactored example with the function "decoupled" from the LeRobot configuration class:


```python
def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, fps: int = 30) -> tuple[dict, bool]:
    """
    Build the feature from the environment without LeRobotDatasetCfg.
    Returns:
        features (dict): The dictionary of feature schemas.
        is_action_aligned (bool): Whether the action dimensions match the joint names.
    """
    features = {}
    default_feature_joint_names = env.cfg.default_feature_joint_names
    
    # 1. Determine Action Dimensions
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    # 2. Logic for Action Alignment
    is_action_aligned = action_dim == len(default_feature_joint_names)
    
    if not is_action_aligned:
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
    else:
        action_joint_names = default_feature_joint_names

    # 3. Build Feature Map
    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names)
    )

    # 4. Handle Cameras
    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = fps  # Passed directly
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features, is_action_aligned
```


## Understanding what to change

To remove **LeRobotDatasetCfg** from the function, we have to replace its role as a "middleman" for configuration. So we need to pass the specific values it was providing from LeRobot (like fps) directly as arguments and handle the **action_align** logic internally or return it as a result.


What changed?

- **Direct Parameter Injection:** Instead of passing the whole **dataset_cfg** object just to get the fps, we now pass fps: int directly. This makes the function easier to test in isolation.

- **Return Values instead of Side-Effects:** Previously, the function modified **dataset_cfg.action_align** inside the body (a side-effect). Now, it calculates **is_action_aligned** and returns it alongside the dictionary. This is generally considered "cleaner" coding practice.

- **Independence:** The function no longer requires the lerobot library's config structures to run, making it more portable if you decide to use a different data logging library later.


**How to call it now:**

If you still need to update a config object elsewhere in your code, you do it at the call site:

```python
# Before
features = build_feature_from_env(env, my_cfg)

# After
features, aligned = build_feature_from_env(env, fps=my_cfg.fps)
my_cfg.action_align = aligned
```

The change to **dataset_cfg.action_align** is not necessary and cause further issues.



## Function Breakdown for build_feature_from_env

The function builds a dictionary of features across three main categories:

**1 - Action Mapping**

It determines the dimensionality of the robot's actions and does the FOllowing

- If the actions match the joints, it uses the joint names.
- If they don't match, it creates generic labels and tells the config that the actions aren't directly aligned with the state.

**2 - Robot State**

It captures the "proprioception" (the internal state of the robot, like joint positions and more). This is mapped to **observation.state** using the default joint names defined in your environment config. Just like IsaacLab.

**3 - Vision (camera stuff)**

The function loops through every camera in your scene and for each one camera, it does the following:

- Extracts the height and width.
- Defines the data type as "video".
- Injects the FPS from the **dataset_cfg**.


The resulting features dictionary will look something like this:

| Key                       | Data Type | Description                                      |
| ------------------------- | --------- | ------------------------------------------------ |
| action                    | float32   | The control signals sent to the robot.           |
| observation.state         | float32   | The actual joint positions/velocities.           |
| observation.images.{name} | video     | Compressed video stream from each camera sensor. |


## What is LeRobotDatasetCfg doing here?

In this specific function, **dataset_cfg** acts as a dynamic configuration tracker. It isn't just a static reference; the function actually modifies it based on what it finds in the environment.

Here are the three specific roles it plays:

- **Flagging Alignment (action_align):** It sets a boolean flag (True or False) within the config. If the number of actions the robot takes matches the number of joint names provided, it assumes they are "aligned." If not (e.g., you're using a high-level controller rather than direct joint control), it marks them as unaligned and generates generic names like dim_0, dim_1.

- **Providing Global Metadata:** It provides the FPS (Frames Per Second) value (**dataset_cfg.fps**). This is critical for the **VideoFeatureItem**, as the dataset needs to know the timing of the recorded frames for proper playback and training.

- **Persisting State:** Since **dataset_cfg** is passed by reference, the changes made inside this function (like setting **action_align**) stick around. This ensures that when the dataset is actually written to disk later, the metadata accurately reflects what was detected in the environment.

**Summary**

This function **build_feature_from_env** is essentially a schema builder. It looks at the simulation environment (the env) and figures out exactly how the data should be structured so that it can be saved into a **LeRobot-compatible dataset**.

**TLDR:** Think of it as a translator that takes the raw physical properties of a robot and turns them into a "metadata map" for training.



# Tracking changes that might have broken the Gr00t

These are the changes as off 02/07/2026, everything works fine

newer version is adding the DROID dataset

Curremtly running:

```toml
"torch==2.7.0",
"torchvision==0.22.0",
```


## Adding DROID

- [DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset](https://droid-dataset.github.io/)
- [nvidia/GR00T-N1.6-DROID](https://huggingface.co/nvidia/GR00T-N1.6-DROID)

examples/DROID/main_gr00t.py

```python
# examples/DROID/main_gr00t.py
# ruff: noqa
# NOTE: this requires installation of the droid repo.
# Adapted from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from collections import deque

import cv2
import imageio
import numpy as np
import pandas as pd
import tqdm
import tyro
from moviepy.editor import ImageSequenceClip
from PIL import Image

from droid.robot_env import RobotEnv
from server_client import PolicyClient
from utils import resize_with_pad

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15
RESOLUTION = (180, 320)  # resize images to this resolution before sending to the policy server


@dataclasses.dataclass
class Args:
    # Hardware parameters

    left_camera_id: str = "<SET THIS>"  # e.g., "24259877"
    right_camera_id: str = "<SET THIS>"  # e.g., "24514023"
    wrist_camera_id: str = "<SET THIS>"  # e.g., "13062452"

    # Policy parameters
    policy_host: str = "localhost"
    policy_port: int = 5555
    policy_api_token: str = None

    results_dir: str = None  # if None, will use the current timestamp as the results directory

    # Rollout parameters
    max_timesteps: int = 600  # how many steps to run each rollout

    # How many actions to execute from a predicted action chunk before querying policy server again
    open_loop_horizon: int = 8
    external_camera: str = (
        "left"  # which exterior camera to use for the policy server, choose from ["left", "right"]
    )
    render_camera: str = "left"  # which camera to render saved video from
    render_fps: int = 50

    debug: bool = False
    vis_cameras: bool = False

    delay_seconds: int = 5


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    assert args.external_camera in ["left", "right"], (
        f"Invalid exterior camera: {args.exterior_camera}"
    )

    if args.results_dir is None:
        results_dir = f"results_gr00t_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    else:
        results_dir = args.results_dir

    # Initialize the Panda environment. N1.6-DROID uses absolute joint position actions.
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    os.makedirs(results_dir, exist_ok=True)

    policy_client = PolicyClient(
        host=args.policy_host, port=args.policy_port, api_token=args.policy_api_token
    )

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    if args.debug:
        debug_dir = os.path.join(results_dir, "debug_data")
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "videos/wrist_image/"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "videos/exterior_image_1_left/"), exist_ok=True)

    instruction = None
    while True:
        if instruction is None:
            instruction = input("Enter instruction: ")
        else:
            if input("Change instruction? (enter y or n) ").lower() == "y":
                instruction = input("Enter instruction: ")

        time.sleep(args.delay_seconds)

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        if args.debug:
            model_wrist_image_writer = imageio.get_writer(
                os.path.join(
                    debug_dir, "videos/wrist_image/", f"model_wrist_image_{timestamp}.mp4"
                ),
                fps=5,
            )
            model_exterior_image_1_left_writer = imageio.get_writer(
                os.path.join(
                    debug_dir,
                    "videos/exterior_image_1_left/",
                    f"model_exterior_image_1_left_{timestamp}.mp4",
                ),
                fps=5,
            )

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        # Profiling variables (reset for each rollout)
        rollout_start_time = time.time()
        obs_times = deque(maxlen=50)  # Track observation collection times
        server_times = deque(maxlen=50)  # Track server response times
        action_count = 0

        for t_step in bar:
            step_start_time = time.time()
            try:
                # Get the current observation
                obs_start_time = time.time()
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )
                obs_time = time.time() - obs_start_time
                obs_times.append(obs_time)

                video.append(curr_obs[f"{args.render_camera}_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if (
                    actions_from_chunk_completed == 0
                    or actions_from_chunk_completed >= args.open_loop_horizon
                ):
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.

                    left_image = resize_with_pad(
                        curr_obs["left_image"], RESOLUTION[0], RESOLUTION[1]
                    )
                    right_image = resize_with_pad(
                        curr_obs["right_image"], RESOLUTION[0], RESOLUTION[1]
                    )
                    wrist_image = resize_with_pad(
                        curr_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1]
                    )

                    if args.external_camera == "left":
                        ext_image = left_image
                    elif args.external_camera == "right":
                        ext_image = right_image

                    if args.debug:
                        model_wrist_image_writer.append_data(wrist_image)
                        model_exterior_image_1_left_writer.append_data(ext_image)

                    request_data = {
                        "video.exterior_image_1_left": ext_image[
                            None, None, ...
                        ],  # [B, T, H, W, C]
                        "video.wrist_image_left": wrist_image[None, None, ...],
                        "state.joint_position": curr_obs["joint_position"][None, None, ...].astype(
                            np.float32
                        ),
                        "state.gripper_position": curr_obs["gripper_position"][
                            None, None, ...
                        ].astype(np.float32),
                        "annotation.language.language_instruction": [instruction],
                    }

                    if args.vis_cameras:
                        # viz the left image 1 and wrist image and use cv2 to display them side by side
                        left_image_display = cv2.resize(
                            left_image, (wrist_image.shape[1], wrist_image.shape[0])
                        )
                        combined_display = np.concatenate([left_image_display, wrist_image], axis=1)
                        # convert to bgr
                        combined_display = combined_display[..., ::-1]
                        cv2.imshow("Camera Views", combined_display)
                        cv2.waitKey(1)

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    server_start_time = time.time()
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [N, 8] of joint position actions (7) + gripper position (1)
                        response = policy_client.get_action(request_data)
                    server_time = time.time() - server_start_time
                    server_times.append(server_time)
                    pred_action_chunk = np.concatenate(
                        (
                            response[0][f"action.joint_position"][0],
                            response[0]["action.gripper_position"][0],
                        ),
                        axis=1,
                    )

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                env.step(action)
                action_count += 1

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - step_start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

                #  profiling stats
                if obs_times:
                    avg_obs_time = np.mean(obs_times) * 1000
                    min_obs_time = np.min(obs_times) * 1000
                    max_obs_time = np.max(obs_times) * 1000
                else:
                    avg_obs_time = min_obs_time = max_obs_time = 0

                if server_times:
                    avg_server_time = np.mean(server_times) * 1000
                    min_server_time = np.min(server_times) * 1000
                    max_server_time = np.max(server_times) * 1000
                else:
                    avg_server_time = min_server_time = max_server_time = 0

                total_elapsed = time.time() - rollout_start_time
                actions_per_sec = action_count / total_elapsed if total_elapsed > 0 else 0

                bar.set_description(
                    f"Obs: {avg_obs_time:.1f}ms [{min_obs_time:.1f}-{max_obs_time:.1f}] | "
                    f"Server: {avg_server_time:.1f}ms [{min_server_time:.1f}-{max_server_time:.1f}] | "
                    f"Actions/sec: {actions_per_sec:.2f}"
                )
            except KeyboardInterrupt:
                break

        os.makedirs(os.path.join(results_dir, "videos"), exist_ok=True)
        video = np.stack(video)
        # replace whitespace with underscores in instruction
        sanitized_instruction = instruction.replace(" ", "_")
        save_filename = os.path.join(
            results_dir, "videos", f"{sanitized_instruction}_video_" + timestamp
        )
        ImageSequenceClip(list(video), fps=args.render_fps).write_videofile(
            save_filename + ".mp4", codec="libx264"
        )

        if args.debug:
            model_wrist_image_writer.close()
            model_exterior_image_1_left_writer.close()

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        new_row = {
            "success": success,
            "duration": t_step,
            "video_filename": save_filename,
        }
        new_index = len(df)
        df.loc[new_index] = new_row

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset(randomize=False)

    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join(results_dir, f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, stereo_camera="left", save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and stereo_camera in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and stereo_camera in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and stereo_camera in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

```

### DROID server_client

examples/DROID/server_client.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import io
from typing import Any

import msgpack
import numpy as np
import zmq


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.
    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)
    Returns:
        JSON-serializable representation of the object
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process the dict
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already JSON-serializable
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # For other types, try to convert to string as fallback
        # You might want to handle specific types differently
        return str(obj)


class MessageType(Enum):
    START_OF_EPISODE = "start_of_episode"
    END_OF_EPISODE = "end_of_episode"
    EPISODE_STEP = "episode_step"
    IMAGE = "image"
    TEXT = "text"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.
    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys."""
    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys."""
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self):
        """Set default values for action-related fields if not specified."""
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_dict"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            # Convert to dict and let msgpack recursively handle nested objects
            return {"__ModalityConfig_class__": True, "as_dict": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class BasePolicy(ABC):
    """Abstract base class for robotic control policies.
    This class defines the interface that all policies must implement, including
    methods for action computation, input/output validation, and state management.
    Subclasses must implement:
        - check_observation(): Validate observation format
        - check_action(): Validate action format
        - _get_action(): Core action computation logic
        - reset(): Reset policy to initial state
    """

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Check if the observation is valid.
        Args:
            observation: Dictionary containing the current state/observation of the environment
        Raises:
            AssertionError: If the observation is invalid.
        """
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        """Check if the action is valid.
        Args:
            action: Dictionary containing the action to be executed
        Raises:
            AssertionError: If the action is invalid.
        """
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation.
        This method should be overridden by subclasses to implement policy-specific
        action computation. Input validation is handled by the public get_action() method.
        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation
        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the action to be executed
                - info: Dictionary containing additional metadata (e.g., confidence scores)
        """
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation with validation.
        This is the main public interface. It validates the observation, calls
        the internal _get_action(), and validates the resulting action.
        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation
        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the validated action
                - info: Dictionary containing additional metadata
        Raises:
            AssertionError/ValueError: If observation or action validation fails
        """
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.
        Args:
            options: Dictionary containing the options for the reset
        Returns:
            Dictionary containing the info after resetting the policy
        """
        pass


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.
        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
```

### DROID utils

examples/DROID/utils.py

```python
"""
Taken from https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client/src/openpi_client
"""

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.
    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.
    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.
    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.
    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
```

### DROID embodiement

Add the new embodiement for DROID at line 303

gr00t/configs/data/embodiment_configs.py

```python
#            modality_keys=["annotation.human.coarse_action"],
#        ),
#    },
    "oxe_droid": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 32)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
            action_configs=[
                # joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
            ],
        ),
    },
}
```

### DROID embodiment_tags

Add at line 47 inside ```gr00t/data/embodiment_tags.py```

```python
    OXE_DROID = "oxe_droid"
    """
    The Open-X-Embodiment DROID robot with relative joint position actions.
    """
```

### Add DROID to processing_gr00t_n1d6

Make changes at line 43, must add "oxe_droid": 16

gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py

```python
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "oxe_droid": 16, # add this to line 43
    "new_embodiment": 10,
}
```


## Changes at GR00T


### Changes to GR00T factory.py fix the GPU issue

Do changes at ```gr00t/data/dataset/factory.py```

```python
from gr00t.experiment.dist_utils import barrier # line 11
```

Replace the following at line 49:

```python
torch.distributed.barrier() 
```

with the following:

```python
barrier()
```

### Changes to GR00T gr00t_policy

Must remove the line 100 at ```gr00t/policy/gr00t_policy.py```

```python
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_delta_indices) == 1, "Only one language delta index is supported" # remove this at line 100
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported" # Add this at line 101
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
```

### Change to GR00T run_gr00t_server

At line 40 inside: ```gr00t/eval/run_gr00t_server.py```

```python
# Server configs
host: str = "127.0.0.1" # removed this
host: str = "0.0.0.0" # addded this
```

### Change to GR00T replay_policy

At the line 311 inside: ```gr00t/policy/replay_policy.py```

```python
# Infer batch size from observation
first_video_key = self.modality_configs["video"].modality_keys[0] # Removed this
batch_size = observation["video"][first_video_key].shape[0] # Removed this
# Added everything below
if observation is not None:
    first_video_key = self.modality_configs["video"].modality_keys[0]
    batch_size = observation["video"][first_video_key].shape[0]
# If batch size is not provided in observation, check if it's provided in options
elif "batch_size" in options:
    batch_size = options["batch_size"]
else:
    batch_size = 1
    print("No batch size provided, using default batch size of 1")
```

### Dependencies

at te ```pyproject.toml```

must upgrade from the following versions at the lines 23 and 24

```toml
"torch==2.7.0",
"torchvision==0.22.0",
```

to the next versions

```toml
"torch==2.7.1",
"torchvision==0.22.1",
```
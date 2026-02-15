# Notes on the dependencies issues for LeIsaac

LeIsaac depends on LeRobot which depends on packages and several versions that causes issues to IsacSim, IsaacLab and many other Nvidia Software.

numpy version is a big one

Error Handling for avoiding Lerobot inside LeIsaac caused infinite loop.



Tracing the Issues:

**Step - 1**

File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 60

```python
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
```

**Step - 2**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/__init__.py", line 11

```python
"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *
from .utils import monkey_patch

```

**Step - 3**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/__init__.py", line 17

```python
"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
```

**Step - 4**

Now starts running Isaaclab stuff.

IsaacLab tasks has an importer util that is recursively importing all the required packages into the Python Wheel so it can leverage the Gym Wrapper to instantiate the envs


File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 40

This line has the Issue:

```python
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass
```


```python
"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    It replaces the need of the following code snippet on the top of each package's ``__init__.py`` file:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass
```


**Step - 5**

File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 84


The line ```__import__(info.name)``` inside

```python
__import__(info.name)
```

Inside here

```python
        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])
```

Part of the Internal Helper

```python
"""
Internal helpers.
"""


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the ``blacklist_pkgs`` argument to skip blacklisted packages. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.

    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)
```


**Step - 6**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/__init__.py", line 1

```python
from .bi_arm_env_cfg import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)
from .direct.bi_arm_env import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
from .direct.single_arm_env import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .lekiwi_env_cfg import (
    LeKiwiActionsCfg,
    LeKiwiEventCfg,
    LeKiwiObservationsCfg,
    LeKiwiRewardsCfg,
    LeKiwiTaskEnvCfg,
    LeKiwiTaskSceneCfg,
    LeKiwiTerminationsCfg,
)
from .single_arm_env_cfg import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
```

**Step - 7**

Here will call a dependency that needs LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader which have LeRobot Implemented because it is made to generate a Dataset compatible with LeRobot.

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/bi_arm_env_cfg.py", line 22

```python
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
```


**Step - 8**

The issue starts by depending on these: LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/__init__.py", line 4


```python
from .device_base import DeviceBase
from .gamepad import SO101Gamepad
from .keyboard import SO101Keyboard
from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader
from .lerobot import BiSO101Leader, SO101Leader
```

**Step - 9**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/__init__.py", line 1

```python
from .lekiwi_gamepad import LeKiwiGamepad
from .lekiwi_keyboard import LeKiwiKeyboard
from .lekiwi_leader import LeKiwiLeader
```

**Step - 10**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/lekiwi_gamepad.py", line 4

```python
from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env
```

**Step - 11**

robot_utils Depends on the Dataset Handler Which is trying to import LeRobot

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12

```python
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
```

**Step - 12**

Here we have the culprit: ModuleNotFoundError: No module named 'lerobot'

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py", line 6

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```



```bash
Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 60, in <module>
    from leisaac.utils.env_utils import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/__init__.py", line 11, in <module>
    from .tasks import *
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/__init__.py", line 17, in <module>
    import_packages(__name__, _BLACKLIST_PKGS)
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 40, in import_packages
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 84, in _walk_packages
    __import__(info.name)
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/__init__.py", line 1, in <module>
    from .bi_arm_env_cfg import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/bi_arm_env_cfg.py", line 22, in <module>
    from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/__init__.py", line 4, in <module>
    from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/__init__.py", line 1, in <module>
    from .lekiwi_gamepad import LeKiwiGamepad
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/lekiwi_gamepad.py", line 4, in <module>
    from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12, in <module>
    from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py", line 6, in <module>
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
ModuleNotFoundError: No module named 'lerobot'
```
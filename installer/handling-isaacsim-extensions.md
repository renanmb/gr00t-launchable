# Handling IsaacSim scripts

Regardless of the method to install the dependencies on IsaacSim (Omniverse kit apps as a whole) you should always version lock it.

Not Version locking can cause all sort of hell.

Solution:

1. Using **omni.kit.pipapi** works




THe issue that if the IsaacSim extension has an extra dependency such as ONNX or ONNXRUNTIME we have to install it directly on IsaacSim Python WHeel.

Using the

```bash
./python.sh -m pip install onnxruntime onnx
```

The Omniverse extensions as descrivbed in the isaaclab have a setup.py to install external dependencies but Omniverse kit apps and IsacSim as such has extra things that might create issues.

```toml
[dependencies]
# 1. Ensure the pip manager loads first
"omni.kit.pipapi" = {} 

[python.pipapi]
# 2. List your required packages
requirements = [
    "onnx",
    "onnxruntime" 
]
# 3. CRITICAL: Allow Omniverse to download from the internet (PyPI)
use_online_index = true
```

We could add the dependency in the extension.toml and hopefully this works


Ensure two things are present in your ```config/extension.toml``` file:

1. A dependency on the pip API: Omniverse needs to load its internal pip manager before it loads your extension.

2. The pip API configuration: This dictates what packages to download and gives permission to go online.


Further Insights (**Troubleshooting**)

If you already had a similar setup and it wasn't working, it is likely due to one of these common Omniverse quirks:

- Missing use_online_index = true: By default, Omniverse tries to look for local pre-bundled .whl files. If this flag is missing or set to false, it will never attempt to download ONNX from PyPI.

- The "Silent Cache" Failure: If an installation failed midway (e.g., due to a timeout or a missing index flag), Omniverse writes to a .install_cache.json file in its hidden pip3-envs folder. Once that file is updated, Omniverse thinks it has already handled the package and will not try again, even if you fix your extension.toml.

    - The Fix: You can force Omniverse to clear its cache and retry by launching Isaac Sim from your terminal and appending the --clear-data flag to the launch command.

- No App Restart: You must completely close and restart the Isaac Sim application after modifying the extension.toml. Simply disabling and re-enabling the extension in the UI is usually not enough to trigger the pip installer.

Failed attempt

```python
# Auto install onnxruntime
import subprocess
import sys

def install_onnx():
    # Use sys.executable to ensure we use the Python inside python.sh
    # We also clear PYTHONPATH temporarily to stop it from looking at your Conda folders
    env = os.environ.copy()
    env.pop("PYTHONPATH", None) 
    
    print("Installing onnxruntime specifically for Isaac Sim...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "onnxruntime"],
            env=env
        )
        print("Installation finished.")
    except Exception as e:
        print(f"Failed to install: {e}")

try:
    import onnxruntime as ort
except ImportError:
    install_onnx()
    # Force a re-search of the site-packages after installation
    import importlib
    import site
    importlib.reload(site)
    import onnxruntime as ort
# Auto install onnxruntime
```

Trying again:

```python
# Auto install onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    import omni.kit.pipapi
    omni.kit.pipapi.install(
        package="onnxruntime",
        version="1.20.0", # Specify the version to ensure stability
        ignore_cache=False,
        use_online_index=True
    )
    import onnxruntime as ort
# Auto install onnxruntime
```


## Using Setuptools like

The IsaacLab Documentation comments about using Setuptools to install any outsdanding Python package.

- [Extension Development](https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/development.html)


Each extension in Isaac Lab is written as a python package and follows the following structure:

```txt
<extension-name>
├── config
│   └── extension.toml
├── docs
│   ├── CHANGELOG.md
│   └── README.md
├── <extension-name>
│   ├── __init__.py
│   ├── ....
│   └── scripts
├── setup.py
└── tests
```

The ```config/extension.toml``` file contains the metadata of the extension. 

When an extension is enabled the python module specified in the ```config/extension.toml``` file is loaded and scripts that contain children of the ```omni.ext.IExt``` class are executed.

While loading extensions into Omniverse happens automatically, using the python package in standalone applications requires additional steps. 

To simplify the build process and avoid the need to understand the **premake** build system used by Omniverse, we directly use the **setuptools** python package to build the python module provided by the extensions. 

This is done by the ```setup.py``` file in the extension directory.

> [!NOTE]  
> The ```setup.py``` file is not required for extensions that are only loaded into Omniverse using the Extension Manager.

This is what I did not quite understood. Why the Extension Manager is not handling the Package ?

**Example of Setup.py**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This needs to be done
Installation script for the 'isaaclab' python package.
"""

import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "onnx==1.16.1",  # 1.16.2 throws access violation on Windows
]

# # Additional dependencies that are only available on Linux platforms
# if platform.system() == "Linux":
#     INSTALL_REQUIRES += [
#         "pin-pink==3.1.0",  # required by isaaclab.isaaclab.controllers.pink_ik
#         "dex-retargeting==0.4.6",  # required by isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils
#     ]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Installation operation
setup(
    name="leatherbackpolicyexample",
    author="Pappachuck",
    maintainer="Jesus",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    # dependency_links=PYTORCH_INDEX_URL,
    packages=["leatherback.policy.example"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
```


## Omniverse Kit Doc recommendations

According to the documentation for Omniverse Kit, they mention a different approach to install the Python Packages.

- [Using Python pip Packages](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/using_pip_packages.html)

There are 2 ways to use python pip packages in extensions:

- Install them at runtime and use them right away.

- Install them at build-time and pre-package into another extension.

### Runtime installation using omni.kit.pipapi

Installing at runtime is probably the most convenient way to quickly prototype and play around with tools. 

You just have to declare a dependency on this extension, and call the ```omni.kit.pipapi.install("package_name")()``` function. 

This method has issues:

- Can be slow. Usually it ends up blocking the process, waiting for the download and installation upon first startup.
- Requires internet access and pip index availability.
- Has security and licensing implications

Even if the version is locked, which should always be the case, the package can still become unavailable at some point, or the content could change.

Example 1

```python
# Package name and module can be different (although rarely is in practice):
import omni.kit.pipapi
omni.kit.pipapi.install("Pillow", module="PIL")
import PIL
```

Example 2

```python
# PIP/Install pip package

# omni.kit.pipapi extension is required
import omni.kit.pipapi

# It wraps `pip install` calls and reroutes package installation into user specified environment folder.
# That folder is added to sys.path.
# Note: This call is blocking and slow. It is meant to be used for debugging, development. For final product packages
# should be installed at build-time and packaged inside extensions.
omni.kit.pipapi.install(
    package="semver",
    version="2.13.0",
    module="semver", # sometimes module is different from package name, module is used for import check
    ignore_import_check=False,
    ignore_cache=False,
    use_online_index=True,
    surpress_output=False,
    extra_args=[]
)

# use
import semver
ver = semver.VersionInfo.parse('1.2.3-pre.2+build.4')
print(ver)
```


### Build-time installation using repo_build

The recommended way to install pip packages is at build time, and to embed it into an extension. 

The **repo_build** tool, when run, can process special config file(s). 

By default: ```deps/pip.toml```. 

Here you can specify a list of packages to install and a target folder:

```toml
[[dependency]]
packages = [
    "watchdog==0.10.4",
]
target = "../_build/target-deps/pip_prebundle"
```

> [!IMPORTANT]
> The version must be specified (locked).

The **repo_build** tool performs an installation only once. 

It then hashes the whole config file and uploads the installed packages into packman. 

On the next launch, it will download the uploaded package from packman, instead of using the pip index. 




Finally, in the **extension.toml**, we need to make sure that this folder is added to ```sys.path```.

```toml
# Demonstrate how to add another folder to sys.path. Here we add pip packages installed at build time.
[[python.module]]
path = "pip_prebundle"
```









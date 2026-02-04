# Notes for installing LeIsaac

The installation required gcc-13

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-13 g++-13
```

Verify installation, since it is experimental build it is on gcc-13 

```bash
gcc-13 --version
```

Setting GCC 13 as Default (Optional)

If you need to make GCC 13 the default gcc command, you can use update-alternatives to configure it: 

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 --slave /usr/bin/g++ g++ /usr/bin/g++-13
```


Installing Build Essential

For a complete development environment (including make, libc, etc.), it is recommended to also install build-essential: 

```bash
sudo apt install build-essential
```

Other Issue

Issue gives me an error that CXXABI_1.3.15 is not found libstdcxx-ng>=13


Install or upggrade the libstdcxx-ng

```bash
conda install -c conda-forge "libstdcxx-ng>=13"
```

verify installation 

```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep CXXABI
```

Force Conda Env Library with LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

there are also other ways like making a symbolic link or using LD_RELOAD

## IMPORTANT must download Assets

[leisaac-example-scene](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)

[kitchen_with_orange.zip](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip)
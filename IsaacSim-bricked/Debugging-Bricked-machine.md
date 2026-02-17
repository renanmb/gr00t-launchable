# Notes on how to debugg the bricked machine


Running the compatibility checker script:

```bash
./isaac-sim.compatibility_check.sh
```

Outputs message:

IOMMU Enabled

An input-output memory management unit (IOMMU) appears to be enabled on this system.

On bare-metal Linux systems, CUDA and the display driver do not support IOMMU-enabled PCIe peer to peer memory copy.

If you are on a bare-metal Linux system, please disable the IOMMU. Otherwise you risk image corruption and program instability.

This typically can be controlled via BIOS settings (Intel Virtualization Technology for Directed I/O (VT-d) or AMD I/O Virtualization Technology (AMD-Vi)) and kernel parameters (iommu, intel_iommu, amd_iommu).

Note that in virtual machines with GPU pass-through (vGPU) the IOMMU needs to be enabled.

Since we can not reliably detect whether this system is bare-metal or a virtual machine, we show this warning in any case when an IOMMU appears to be enabled.


Also, might want to turn off iommu

```bash
sudo vim /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=off"
sudo update-grub
sudo reboot
```
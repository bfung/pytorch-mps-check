#!/usr/bin/env python

# Scan the PyObjc documentation
# Read the Metal documentation, flip it from Swift to Objective-C
# do a lot of trial and error on accessing names.

from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict

import subprocess 

import Metal
import MetalPerformanceShaders

import torch

MTLGPUFAMILY = {
    "MTLGPUFamilyApple1": Metal.MTLGPUFamilyApple1,
    "MTLGPUFamilyApple2": Metal.MTLGPUFamilyApple2,
    "MTLGPUFamilyApple3": Metal.MTLGPUFamilyApple3,
    "MTLGPUFamilyApple4": Metal.MTLGPUFamilyApple4,
    "MTLGPUFamilyApple5": Metal.MTLGPUFamilyApple5,
    "MTLGPUFamilyApple6": Metal.MTLGPUFamilyApple6,
    "MTLGPUFamilyApple7": Metal.MTLGPUFamilyApple7,
    "MTLGPUFamilyCommon1": Metal.MTLGPUFamilyCommon1,
    "MTLGPUFamilyCommon2": Metal.MTLGPUFamilyCommon2,
    "MTLGPUFamilyCommon3": Metal.MTLGPUFamilyCommon3,
    "MTLGPUFamilyMac1": Metal.MTLGPUFamilyMac1,
    "MTLGPUFamilyMac2": Metal.MTLGPUFamilyMac2,
    "MTLGPUFamilyMacCatalyst1": Metal.MTLGPUFamilyMacCatalyst1,
    "MTLGPUFamilyMacCatalyst2": Metal.MTLGPUFamilyMacCatalyst2
}

@dataclass
class MTLDeviceGpuInfo:
    device: Any
    name: str
    is_default: bool
    has_mps_support: bool
    gpu_family_support: Dict[str, bool]


if __name__ == "__main__":
    devices = Metal.MTLCopyAllDevices()
    num_devices = len(devices)
    default_mtl_device = Metal.MTLCreateSystemDefaultDevice()

    gpu_infos = []
    for idx, device in enumerate(devices): 
        gpu_family_support = {}
        for gpufamily, enum in MTLGPUFAMILY.items():
            gpu_family_support[gpufamily] = device.supportsFamily_(enum)

        gpu_info = MTLDeviceGpuInfo(
            device,
            device.name(),
            device == default_mtl_device,
            MetalPerformanceShaders.MPSSupportsMTLDevice(device),
            gpu_family_support
        )
        gpu_infos.append(gpu_info)
    pprint(gpu_infos)
    print()

    try:
        print("Checking pytorch support...")
        mps_device = torch.device("mps")
        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        print("You're good to go, most likely.")
    except RuntimeError as re:
        print(re)
        
    print()
    print("Running `sw_vers`...")
    subprocess.run(["sw_vers"])
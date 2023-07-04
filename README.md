# GPU benchsuites 
Compile script written by **Jong Hyun Jeong**
**emali: dida1245@korea.ac.kr**
**email: dida1245@gmail.com**

This is a repository of GPU benchmarks for [AccelSim](https://github.com/accel-sim/accel-sim-framework) and [GPGPU-Sim v4.2](https://github.com/accel-sim/gpgpu-sim_distribution). It is based on [gpu-app-collection](https://github.com/accel-sim/gpu-app-collection) and updated for estimating and simulating the state of the art GPU architectures (Volta, Turing, Ampere). It supports **compute capability: 70 75 86**.

Note that needs to be **compiled anew** to run benchmarks on GPGPU-Sim v4.2. Some benchmarks are inherently unexecutable on GPGPU-Sim. Preferably, run the compiled binary on AccelSim. See [here](https://github.com/KU-CSArch/accel-sim_dev) for more information on how to use AccelSim.

## Environment settings
* ubuntu-18.04
* gcc-7.5.0
* cuda-11.3
* cudnn-8.2.0 (include installation script)
* cmake-3.15.1 (include installation script)

## Benchsuites list 
* [Rodinia-3.1](https://github.com/yuhc/gpu-rodinia)
* [Polybench-3.2](https://github.com/cavazos-lab/PolyBench-ACC)
* [Parboil](https://github.com/abduld/Parboil)
* [Ispass-2009](https://github.com/gpgpu-sim/ispass2009-benchmarks)
* [Tango](https://gitlab.com/Tango-DNNbench/Tango)
* [graphBIG](https://github.com/graphbig/graphBIG)
* [Cutlass-1.3.1](https://github.com/NVIDIA/cutlass) **only for arch_86**
* [DeepBench](https://github.com/baidu-research/DeepBench)
* [Gardenia](https://github.com/chenxuhao/gardenia)
* [gSuite](https://github.com/tekdogan/gsuite) **only for arch_86**
* LonestarGPU-2.0 (old version)
* [LonestarGPU-6.0](https://github.com/IntelligentSoftwareSystems/Galois)
* [GNNMark](https://gitlab.com/GNNMark/gnnmark/-/tree/master/) 
* CUDA Samples (not yet)
* MLperf (not yet)

## Usage
Note that needs [cuDNN8.2.0 installer](https://developer.nvidia.com/rdp/cudnn-archive) for **compiling DeepBench**. Select Download **cuDNN v8.2.0 (April 23rd, 2021), for CUDA 11.x** and download cuDNN Library for **Linux (x86_64)**. And then, locate **tar file** to **$(GPUAPPS_ROOT)**. If it is not necessary to compiling DeepBench, ignore here.

Note that needs to create conda environment for running GNNMark. See **gnnmark_env.txt**. 

	$ ./install_dependency.sh 
	$ source setup_environment

**make all**
	$ make

**make selectively**
+ rodinia 
+ lonestar2 
+ polybench 
+ parboil 
+ ispass 
+ deepbench
+ tango 
+ graphbig 
+ lonestar6 
+ cutlass 
+ gardenia 
+ gsuite

**binary path**
	$ GPUAPPS_ROOT/bin/11.3/

## Refence
[AccelSim](https://ieeexplore.ieee.org/document/9138922), ISCA2020
[AccelWattch](https://dl.acm.org/doi/abs/10.1145/3466752.3480063) MICRO2021
[GPGPU-Sim, ISPASS benchsuite](https://ieeexplore.ieee.org/document/4919648) ISPASS2009
[Rodinia]
[Parboil]
[Polybench]
[LonestarGPU]
[gSuite](https://ieeexplore.ieee.org/document/9975401) IISWC2022
[DNNMark](https://ieeexplore.ieee.org/document/9408205) ISPASS2021
[Tango](https://dl.acm.org/doi/10.1145/3300053.3319418) GPGPU19
[Cutlass]
[DeepBench]
[graphBig](https://ieeexplore.ieee.org/document/7832843) SC15
[Gargenia](https://dl.acm.org/doi/10.1145/3283450) ACM Journal on Emerging Technologies in Computing Systems, 2019
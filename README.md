# GPU benchsuites 
compile script written by **Jong Hyun Jeong**

compile with **compute capability 70, 75, 86**

### emali: dida1245@korea.ac.kr
### email: dida1245@gmail.com

## Environment settings:
* ubuntu-18.04
* gcc-7.5.0
* cuda-11.3

## Benchsuites list: 
* Rodinia-3.1
* Polybench-3.2
* Parboil
* Ispass-2009
* CUDA SDK
* LonestarGPU-2.0
* Tango
* graphBIG
* Cutlass-1.3.1 (only SM86)
* DeepBench
* Gardenia
* LonestarGPU-6.0
* DNNMark (not yet)
* GNNMark (not yet)
* MLperf (not yet)
* gSuite (not yet)
* cuGraph (not yet) 
* nvgraph (not yet)

## Usage:
* need [cuDNN8.2.0 installer](https://developer.nvidia.com/rdp/cudnn-archive) for compiling DeepBench
* Select Download cuDNN v8.2.0 (April 23rd, 2021), for CUDA 11.x
* Download cuDNN Library for Linux (x86_64)
* Locate tar file to $(GPUAPPS_ROOT)
	
	./install_dependency.sh 

	source setup_environment

	make
	
**binary path** : $(GPUAPPS_ROOT)/bin/11.3/

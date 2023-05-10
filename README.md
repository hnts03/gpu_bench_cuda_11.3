GPU benchsuites compile script written by Jong Hyun Jeong

emali: dida1245@korea.ac.kr
email: dida1245@gmail.com

Environment settings:

	ubuntu-18.04
	gcc-7.5.0
	cuda-11.3
	RTX 3070

Benchsuites list:

	Rodinia-3.1
	Polybench-3.2
	Parboil
	Ispass-2009
	CUDA SDK
	LonestarGPU-2.0
	Tango
	graphBIG

Accel-SIM only:

	Cutlass-1.3.1
	DeepBench
	Gardenia
	LonestarGPU-6.0
	DNNMark
	GNNMark
	MLperf
	gSuite
	cuGraph

Usage:

	./install_dependency.sh (need cuDNN8.5.0 installer for compiling DeepBench)
		https://developer.nvidia.com/rdp/cudnn-archive
		Select cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x
		Download Local Installer for Linux x86_64 (Tar)
		Locate tar.xz file to $(GPUAPPS_ROOT)

	source setup_environment
	*** get datasets ***
	make
	
	binary path: $(GPUAPPS_ROOT)/bin/11.3/	

GPU benchsuites

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

	./install_dependency.sh (need cuDNN installer for compiling DeepBench)
	source setup_environment
	*** get datasets ***
	make
	
	binary path: $(GPUAPPS_ROOT)/bin/11.3/	

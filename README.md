<<<<<<< HEAD
GPU benchsuites \
Environment settings \
	ubuntu-18.04 \
	gcc-7.5.0 \
	cuda-11.3 \
	cudnn-8.5.0 (for DeepBench) \
 \
Benchsuites list \
	Rodinia-3.1\
	Polybench-3.2\
	Parboil\
	Ispass-2009\
	CUDA SDK\
	LonestarGPU-6.0\
	Tango\

Accel-SIM only\
	Cutlass-1.3.1\
	DeepBench\
	Gardenia\
	LonestarGPU-6.0\
	GraphBig\
	DNNMark\
	GNNMark\
	MLperf\
	gSuite\
	cuGraph\

Usage:\
	source setup_cuda11.3 (setup_cudnn8.5.0 for DeepBench)\
	source setup_environment\
	*** get datasets ***\
	make\
	\
	binary path: $(GPUAPPS_ROOT)/bin/11.3/	\
=======
GPU benchsuites

Environment settings:

	ubuntu-18.04
	gcc-7.5.0
	cuda-11.3
	cudnn-8.5.0 (for DeepBench)

Benchsuites list:

	Rodinia-3.1
	Polybench-3.2
	Parboil
	Ispass-2009
	CUDA SDK
	LonestarGPU-6.0
	Tango

Accel-SIM only:

	Cutlass-1.3.1
	DeepBench
	Gardenia
	LonestarGPU-6.0
	GraphBig
	DNNMark
	GNNMark
	MLperf
	gSuite
	cuGraph

Usage:

	source setup_cuda11.3 (setup_cudnn8.5.0 for DeepBench)
	source setup_environment
	*** get datasets ***
	make
	
	binary path: $(GPUAPPS_ROOT)/bin/11.3/	
>>>>>>> 08a74819ee5715865bcdb5ecfbffa45d4a87f03c

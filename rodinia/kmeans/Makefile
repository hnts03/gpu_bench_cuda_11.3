include ../../common/make.config

#GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
#GENCODE_SM61 ?= -gencode=arch=compute_61,code=\"sm_75,compute_61\"
#GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"

# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc $(GENCODE_SM75) $(GENCODE_SM70) $(GENCODE_SM61)
NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif



# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

kmeans: kmeans.o cluster.o getopt.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CC_FLAGS) kmeans.o cluster.o getopt.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcudart -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

kmeans.o: kmeans.c
	$(CC) -c kmeans.h kmeans.c

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo

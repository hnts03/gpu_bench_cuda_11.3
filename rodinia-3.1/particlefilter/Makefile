#makefile

include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -lcudart $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM86) $(GENCODE_SM60) $(GENCODE_SM62)

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -g -lm -O3 -use_fast_math  ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -g -lm -O3 -use_fast_math ex_particle_CUDA_float_seq.cu -o particlefilter_float

clean:
	rm -f particlefilter_naive particlefilter_float

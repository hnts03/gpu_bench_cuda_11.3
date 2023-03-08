include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc $(GENCODE_SM75) $(GENCODE_SM70) $(GENCODE_SM86) $(GENCODE_SM60) $(GENCODE_SM62)

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart 

clean:
	rm -f gaussian

include ../../common/make.config
CC := $(CUDA_DIR)/bin/nvcc -lcudart

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

GENCODE_SM10 =
GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM61 ?= -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"



hybridsort: main.cu  bucketsort.cu bucketsort.cuh bucketsort_kernel.cu histogram1024_kernel.cu  mergesort.cu mergesort.cuh mergesort_kernel.cu
	$(CC) $(GENCODE_SM75) $(GENCODE_SM70) $(GENCODE_SM86) $(GENCODE_SM62) $(GENCODE_SM60) $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) bucketsort.cu mergesort.cu main.cu -o hybridsort


clean:
	rm	-f *.o hybridsort

GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"

all:
	nvcc -g -lcublas -lcublasLt -lculibos -ldl -lpthread -lcudart -lcudadevrt -std=c++11 $(GENCODE_SM86) cuBlasUtil.cu Data_Util.cu  CU_GCN_MP.cu CU_SpMM_GCN.cu CU_SpMM_GIN.cu CU_SAG_WL.cu CU_GIN_WL.cu CU_SAG_MP.cu cudaDataLoader.cu scatter_cuda.cu linear.cu index_select.cu main.cpp -o cudaDataLoader
clean:
	rm cudaDataLoader

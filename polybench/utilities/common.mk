GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM62 ?= -gencode=arch=compute_61,code=\"sm_61,compute_61\"


all:
	nvcc $(GENCODE_SM75) $(GENCODE_SM70) $(GENCODE_SM61) ${CUFILES} -I${PATH_TO_UTILS} -lcudart -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe

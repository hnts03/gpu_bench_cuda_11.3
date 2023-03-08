GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"


all:
	nvcc $(GENCODE_SM75) $(GENCODE_SM86) $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM62) ${CUFILES} -I${PATH_TO_UTILS} -lcudart -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe

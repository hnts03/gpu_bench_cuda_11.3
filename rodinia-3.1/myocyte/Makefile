GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
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
# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte:		main.o
	nvcc  $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM86) $(GENCODE_SM60) $(GENCODE_SM62)	 main.o \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm -lcudart \
                                -o myocyte

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	nvcc	 $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM86) $(GENCODE_SM60) $(GENCODE_SM62) main.cu \
				-c -O3 -g

# delete all object files
clean:
	rm -f *.o *.out output.txt

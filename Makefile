GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"
# MAKE_ARGS := $(GENCODE_SM86) $(GENCODE_SM60)  $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)

ifeq ($(GPUAPPS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

ifeq ($(CUDA_GT_10), 1)
all: rodinia lonestar polybench parboil
endif
# ifeq ($(CUDA_GT_7), 1)
# # all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive dragon-cdp microbench rodinia ispass-2009 lonestargpu-2.0 polybench Parboil shoc custom_apps deeplearning cutlass GPU_Microbenchmark heterosync Deepbench_nvidia
# all: rodinia lonestargpu-2.0 polybench Parboil tango
# else
# 	ifeq ($(CUDA_GT_4), 1)
# 	all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive microbench rodinia ispass-2009 dragon-cdp lonestargpu-2.0 polybench Parboil shoc custom_apps
# 	else
# 	all:   pannotia rodinia_2.0-ft proxy-apps microbench rodinia ispass-2009 polybench Parboil shoc custom_apps
# 	endif
# endif

#Disable clean for now, It has a bug!
# clean_dragon-naive clean_pannotia clean_proxy-apps
#clean: clean_rodinia_2.0-ft clean_dragon-cdp  clean_ispass-2009 clean_lonestargpu-2.0 clean_custom_apps clean_Parboil clean_cutlass clean_rodinia clean_heterosync
clean: clean_rodinia clean_lonestar clean_parboil

clean_data:
	./clean_data.sh

data:
	mkdir -p $(BINDIR)/
	cd ../ && bash ./get_data.sh

###################################################################################################3
# Rodinia 2.0 Functional Test Stuff
# ###################################################################################################3
# rodinia_2.0-ft:
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/backprop
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/bfs
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/heartwall
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/hotspot
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/kmeans
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/lud
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/nn
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/nw
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/pathfinder
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/srad
# 	$(SETENV) make $(MAKE_ARGS) -C rodinia-3.1/2.0-ft/streamcluster

# ###################################################################################################3
# # Purdue microbenchmarks for added functionality
# ###################################################################################################3
# microbench:
# 	$(SETENV) make $(MAKE_ARGS) -C microbench cuda-$(CUDA_VERSION_MAJOR)

# ###################################################################################################3
# # For Dragon, we need to change the archs manually!  (TO DO)
# # Naive works wwith only SM_20 and above
# # Fro Dragon-cdp comilation, you need to ensure that you are using at least CUDA 5.5
# ###################################################################################################3
# dragon-naive: 
# 	chmod +x dragon_li/sconstruct
# 	if [ ${CUDA_VERSION_MAJOR} -lt 8 ]; then \
# 		scons sm35=1 no_debug=1 -C dragon_li; \
# 	else \
# 		scons sm35=1 sm61=1 no_debug=1 -C dragon_li; \
# 	fi
# 	cp ./dragon_li/bin/$(CUDA_VERSION)/testAmr $(BINDIR)/
# 	cp ./dragon_li/bin/$(CUDA_VERSION)/testBfs $(BINDIR)/
# 	cp ./dragon_li/bin/$(CUDA_VERSION)/testJoin $(BINDIR)/
# 	cp ./dragon_li/bin/$(CUDA_VERSION)/testSssp $(BINDIR)/

# dragon-cdp: dragon-naive
# 	chmod +x dragon_li/sconstruct
# 	if [ ${CUDA_VERSION_MAJOR} -lt 8 ]; then \
# 		scons cdp=1 no_debug=1 sm35=1 -C dragon_li; \
# 	else \
# 		scons cdp=1 no_debug=1 sm61=1 sm35=1 -C dragon_li; \
# 	fi
# 	cp ./dragon_li/cdp_bin/$(CUDA_VERSION)/testAmr-cdp $(BINDIR)/
# 	cp ./dragon_li/cdp_bin/$(CUDA_VERSION)/testBfs-cdp $(BINDIR)/
# 	cp ./dragon_li/cdp_bin/$(CUDA_VERSION)/testJoin-cdp $(BINDIR)/
# 	cp ./dragon_li/cdp_bin/$(CUDA_VERSION)/testSssp-cdp $(BINDIR)/

# ###################################################################################################3
# #Microbenchmarks for cache
# ###################################################################################################3

# GPU_Microbenchmark:
# 	mkdir -p $(BINDIR)/
# 	$(SETENV) make $(MAKE_ARGS) -C GPU_Microbenchmark
# 	cp -r GPU_Microbenchmark/bin/* $(BINDIR)/


Deepbench_nvidia:
	$(SETENV) make $(MAKE_ARGS) -C DeepBench/code/nvidia
	cp -r DeepBench/code/nvidia/bin/conv_bench* $(BINDIR)/
	cp -r DeepBench/code/nvidia/bin/gemm_bench* $(BINDIR)/
	cp -r DeepBench/code/nvidia/bin/rnn_bench* $(BINDIR)/

###################################################################################################3
#pagerank and bc does not work with SM_10 because they need atomic_add
###################################################################################################3

# pannotia:
# 	$(SETENV) make $(MAKE_ARGS) -C pannotia/bc
# 	$(SETENV) export VARIANT="MAX"; make $(MAKE_ARGS) -C pannotia/color
# 	$(SETENV) export VARIANT="MAXMIN"; make $(MAKE_ARGS) -C pannotia/color
# 	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) -C pannotia/fw
# 	$(SETENV) export VARIANT="BLOCK"; make $(MAKE_ARGS) -C pannotia/fw
# 	$(SETENV) make $(MAKE_ARGS) -C pannotia/mis
# 	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) -C pannotia/pagerank
# 	$(SETENV) export VARIANT="SPMV"; make $(MAKE_ARGS) -C pannotia/pagerank
# 	$(SETENV) export VARIANT="CSR"; make $(MAKE_ARGS) -C pannotia/sssp
# 	$(SETENV) export VARIANT="ELL"; make $(MAKE_ARGS) -C pannotia/sssp

###################################################################################################3
#TO DO
#note: matvec does not work with cuda 8.0
#comd does not wark with cuda 4.2
#xsbench does not work with SM_10
###################################################################################################3


# proxy-apps:
# 	chmod +x proxy-apps-doe/cns/compile.bash
# 	($(SETENV) cd proxy-apps-doe/cns/ ; ./compile.bash)
# 	#chmod +x proxy-apps-doe/comd/cmd_compile.sh
# 	#( cd proxy-apps-doe/comd ; ./cmd_compile.sh)
# 	$(SETENV) make $(MAKE_ARGS) -C proxy-apps-doe/lulesh
# 	if [ ${CUDA_VERSION_MAJOR} -lt 7 ] ; then  \
# 		$(SETENV) make $(MAKE_ARGS) -C proxy-apps-doe/minife_matvec_ell;\
# 	fi
# 	$(SETENV) make $(MAKE_ARGS) -C proxy-apps-doe/xsbench

rodinia:
	mkdir -p $(BINDIR)/
	if [ ${CUDA_VERSION_MAJOR} -gt 5 ]; then \
		$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/huffman/; \
	fi
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/backprop
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/bfs 
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/cfd
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hotspot 
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/kmeans 
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/needle 
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/streamingcluster
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/mummergpu
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/b+tree/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/dwt2d/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/heartwall/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hybridsort/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/myocyte/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/nn/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/particlefilter/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/pathfinder/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/lavaMD/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/lud/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/leukocyte/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hotspot3D/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/gaussian/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/srad/
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/srad/srad_v2 -f Makefile_nvidia
	if [ ${CUDA_VERSION_MAJOR} -gt 5 ]; then \
		mv rodinia-3.1/huffman/huffman $(BINDIR)/huffman; \
	fi
	mv rodinia-3.1/b+tree/b+tree $(BINDIR)/b+tree
	mv rodinia-3.1/dwt2d/dwt2d $(BINDIR)/dwt2d
	mv rodinia-3.1/heartwall/heartwall $(BINDIR)/heartwall
	mv rodinia-3.1/hybridsort/hybridsort $(BINDIR)/hybridsort
	mv rodinia-3.1/myocyte/myocyte $(BINDIR)/myocyte
	mv rodinia-3.1/nn/nn $(BINDIR)/nn
	mv rodinia-3.1/particlefilter/particlefilter_float $(BINDIR)/particlefilter_float
	mv rodinia-3.1/particlefilter/particlefilter_naive $(BINDIR)/particlefilter_naive
	mv rodinia-3.1/pathfinder/pathfinder $(BINDIR)/pathfinder
	mv rodinia-3.1/lavaMD/lavaMD $(BINDIR)/lavaMD
	mv rodinia-3.1/lud/lud $(BINDIR)/lud
	mv rodinia-3.1/leukocyte/leukocyte $(BINDIR)/leukocyte
	mv rodinia-3.1/hotspot3D/hotspot3D $(BINDIR)/hotspot3D
	mv rodinia-3.1/gaussian/gaussian $(BINDIR)/gaussian
	mv rodinia-3.1/srad/srad_v1/srad1 $(BINDIR)/srad1
	mv rodinia-3.1/srad/srad_v2/srad2 $(BINDIR)/srad2
	mv rodinia-3.1/backprop/backprop $(BINDIR)/backprop
	mv rodinia-3.1/bfs/bfs  $(BINDIR)/bfs
	mv rodinia-3.1/cfd/euler3d $(BINDIR)/euler3d
	mv rodinia-3.1/cfd/pre_euler3d $(BINDIR)/pre_euler3d
	mv rodinia-3.1/cfd/euler3d_double $(BINDIR)/euler3d_double
	mv rodinia-3.1/cfd/pre_euler3d_double $(BINDIR)/pre_euler3d_double
	mv rodinia-3.1/hotspot/hotspot $(BINDIR)/hotspot
	mv rodinia-3.1/kmeans/kmeans $(BINDIR)/kmeans
	mv rodinia-3.1/needle/needle $(BINDIR)/nw
	mv rodinia-3.1/streamingcluster/streamingcluster $(BINDIR)/streamingcluster
#	mv $(BINDIR)/mummergpu $(BINDIR)/mummergpu

# ispass-2009:
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/AES
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/BFS
# #	cd CP; export Parboil_ROOT=`pwd`; cd common/src; make $(MAKE_ARGS); cd -; ./Parboil compile cp cuda_short; cp benchmarks/cp/build/cuda_short/cp $(BINDIR)/CP 
# #	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG/3rdParty/ParMetis
# #	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/LIB
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/LPS
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/MUM
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/NN
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/NQU
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/RAY
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/STO
# 	PID=$$$$ && cp -r ispass-2009/WP ispass-2009/WP-$$PID && $(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C ispass-2009/WP-$$PID && rm -rf ispass-2009/WP-$$PID

lonestar:
	$(setenv) make $(make_args) noinline=$(noinline) -C lonestargpu-2.0 all
	mv lonestargpu-2.0/apps/bfs/bfs $(BINDIR)/lonestar_bfs
	mv lonestargpu-2.0/apps/bfs/bfs-atomic $(BINDIR)/lonestar_bfs-atomic
	mv lonestargpu-2.0/apps/bfs/bfs-wlc $(BINDIR)/lonestar_bfs-wlc
	mv lonestargpu-2.0/apps/bfs/bfs-wla $(BINDIR)/lonestar_bfs-wla
	mv lonestargpu-2.0/apps/bfs/bfs-wlw $(BINDIR)/lonestar_bfs-wlw
	mv lonestargpu-2.0/apps/bh/bh $(BINDIR)/bh
	mv lonestargpu-2.0/apps/dmr/dmr $(BINDIR)/dmr
	mv lonestargpu-2.0/apps/mst/mst $(BINDIR)/mst
	mv lonestargpu-2.0/apps/pta/pta $(BINDIR)/pta
	mv lonestargpu-2.0/apps/nsp/nsp $(BINDIR)/nsp
	mv lonestargpu-2.0/apps/sssp/sssp $(BINDIR)/sssp
	mv lonestargpu-2.0/apps/sssp/sssp-wlc $(BINDIR)/sssp-wlc
	mv lonestargpu-2.0/apps/sssp/sssp-wln $(BINDIR)/sssp-wln

parboil:
#	make data
	mkdir -p $(BINDIR)/
	$(SETENV) cd Parboil; ./parboil compile cutcp cuda
	$(SETENV) cd Parboil; ./parboil compile bfs cuda
	$(SETENV) cd Parboil; ./parboil compile histo cuda
	$(SETENV) cd Parboil; ./parboil compile lbm cuda
	$(SETENV) cd Parboil; ./parboil compile mri-gridding cuda
	$(SETENV) cd Parboil; ./parboil compile mri-q cuda
	$(SETENV) cd Parboil; ./parboil compile sad cuda
	$(SETENV) cd Parboil; ./parboil compile sgemm cuda
	$(SETENV) cd Parboil; ./parboil compile spmv cuda
	$(SETENV) cd Parboil; ./parboil compile stencil cuda
	$(SETENV) cd Parboil; ./parboil compile tpacf cuda
	mv ./Parboil/benchmarks/lbm/build/cuda_default/lbm $(BINDIR)/lbm
	mv ./Parboil/benchmarks/cutcp/build/cuda_default/cutcp $(BINDIR)/cutcp
	mv ./Parboil/benchmarks/bfs/build/cuda_default/bfs $(BINDIR)/parboil_bfs
	mv ./Parboil/benchmarks/histo/build/cuda_default/histo $(BINDIR)/histo
	mv ./Parboil/benchmarks/mri-gridding/build/cuda_default/mri-gridding $(BINDIR)/mri-gridding
	mv ./Parboil/benchmarks/mri-q/build/cuda_default/mri-q $(BINDIR)/mri-q
	mv ./Parboil/benchmarks/sad/build/cuda_default/sad $(BINDIR)/sad
	mv ./Parboil/benchmarks/sgemm/build/cuda_default/sgemm $(BINDIR)/sgemm
	mv ./Parboil/benchmarks/spmv/build/cuda_default/spmv $(BINDIR)/spmv
	mv ./Parboil/benchmarks/stencil/build/cuda_default/stencil $(BINDIR)/stencil
	mv ./Parboil/benchmarks/tpacf/build/cuda_default/tpacf $(BINDIR)/tpacf

polybench:
	mkdir -p $(BINDIR)/
	$(SETENV) cd polybench-3.2/; sh ./compileCodes.sh bin
	cp polybench-3.2/bin/* $(BINDIR)/
	rm -r polybench-3.2/bin
# mv polybench/convolution-2d/2DConvolution.exe $(BINDIR)/2DConvolution
# mv polybench/2MM/2mm.exe $(BINDIR)/2mm
# mv polybench/3DCONV/3DConvolution.exe $(BINDIR)/3DConvolution
# mv polybench/3MM/3mm.exe $(BINDIR)/3mm
# mv polybench/ATAX/atax.exe $(BINDIR)/atax
# mv polybench/BICG/bicg.exe $(BINDIR)/bicg
# mv polybench/CORR/correlation.exe $(BINDIR)/correlation
# mv polybench/COVAR/covariance.exe $(BINDIR)/covariance
# mv polybench/FDTD-2D/fdtd2d.exe $(BINDIR)/fdtd2d
# mv polybench/GEMM/gemm.exe $(BINDIR)/gemm
# mv polybench/GESUMMV/gesummv.exe $(BINDIR)/gesummv
# mv polybench/GRAMSCHM/gramschmidt.exe $(BINDIR)/gramschmidt
# mv polybench/MVT/mvt.exe $(BINDIR)/mvt
# mv polybench/SYR2K/syr2k.exe $(BINDIR)/syr2k
# mv polybench/SYRK/syrk.exe $(BINDIR)/syrk

# shoc:
# 	mkdir -p $(BINDIR)/
# 	cd shoc-master/; ./configure; $(SETENV) make $(MAKE_ARGS); $(SETENV) make $(MAKE_ARGS) -C src/cuda
# 	mv shoc-master/src/level0/BusSpeedDownload $(BINDIR)/shoc-BusSpeedDownload
# 	mv shoc-master/src/level0/BusSpeedReadback $(BINDIR)/shoc-BusSpeedReadback
# 	mv shoc-master/src/level0/DeviceMemory $(BINDIR)/shoc-DeviceMemory
# 	mv shoc-master/src/level0/MaxFlops $(BINDIR)/shoc-MaxFlops
# 	mv shoc-master/src/level1/bfs/BFS $(BINDIR)/shoc-BFS
# 	mv shoc-master/src/level1/fft/FFT $(BINDIR)/shoc-FFT
# 	mv shoc-master/src/level1/gemm/GEMM $(BINDIR)/shoc-GEMM
# 	mv shoc-master/src/level1/md/MD $(BINDIR)/shoc-MD
# 	mv shoc-master/src/level1/md5hash/MD5Hash $(BINDIR)/shoc-MD5Hash
# 	mv shoc-master/src/level1/neuralnet/NeuralNet $(BINDIR)/shoc-NeuralNet
# 	mv shoc-master/src/level1/reduction/Reduction $(BINDIR)/shoc-Reduction
# 	mv shoc-master/src/level1/scan/Scan $(BINDIR)/shoc-Scan
# 	mv shoc-master/src/level1/sort/Sort $(BINDIR)/shoc-Sort
# 	mv shoc-master/src/level1/spmv/Spmv $(BINDIR)/shoc-Spmv
# 	mv shoc-master/src/level1/stencil2d/Stencil2D $(BINDIR)/shoc-Stencil2D
# 	mv shoc-master/src/level1/triad/Triad $(BINDIR)/shoc-Triad
# 	mv shoc-master/src/level2/qtclustering/QTC $(BINDIR)/shoc-QTC
# 	mv shoc-master/src/level2/s3d/S3D $(BINDIR)/shoc-S3D
# 	mv shoc-master/src/stability/Stability $(BINDIR)/shoc-Stability

# custom_apps:
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C custom-apps/shoc-modified-spmv/
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C custom-apps/rodinia-kmn-no-tex/
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C custom-apps/sdk-matrixMul-modified/

# power:
# 	$(SETENV) make $(MAKE_ARGS) PWRTYPE=SM noinline=$(noinline) -C gpuwattch-ubench/ power
# 	$(SETENV) make $(MAKE_ARGS) PWRTYPE=HW noinline=$(noinline) -C gpuwattch-ubench/ power

# deeplearning:
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cudnn/mnist
# 	cp cudnn/mnist/mnistCUDNN $(BINDIR)/

cutlass:
	mkdir -p $(BINDIR)/
	git submodule init && git submodule update
	$(SETENV) mkdir -p cutlass-bench/build && cd cutlass-bench/build && cmake .. -DUSE_GPGPUSIM=1 -DCUTLASS_NVCC_ARCHS=70 && make cutlass_perf_test
	cd cutlass-bench/build/tools/test/perf && ln -s -f ../../../../binary.sh . && ./binary.sh
	cp cutlass-bench/build/tools/test/perf/cutlass_perf_test $(BINDIR)/

# Maybe we should use submodules for this - but I have heard a lot of horor stories about these..
# For now - lets just clone if we don't have it and set the SHA we want.
# heterosync:
# 	mkdir -p $(BINDIR)/
# 	cd cuda && \
# 	if [ ! -d "heterosync" ]; then \
# 		git clone git@github.com:mattsinc/heterosync.git; \
# 	fi && \
# 	cd heterosync && git checkout 22bc0eb
# 	$(SETENV) make $(MAKE_ARGS) CUDA_DIR=$(CUDA_INSTALL_PATH) -C heterosync/syncPrims/uvm/
# 	mv heterosync/syncPrims/uvm/allSyncPrims-1kernel $(BINDIR)/

# clean_heterosync:
# 	rm -rf heterosync

clean_cutlass:
	rm -fr cutlass-bench/build

# clean_deeplearning:
# 	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cudnn/mnist clean

# clean_custom_apps:
# 	make clean -C custom-apps/shoc-modified-spmv/
# 	make clean -C custom-apps/rodinia-kmn-no-tex/
# 	make clean -C custom-apps/sdk-matrixMul-modified/

# clean_shoc:
# 	cd shoc-master/; make clean; make distclean

clean_parboil:
	$(SETENV) cd Parboil; ./parboil clean cutcp cuda
	$(SETENV) cd Parboil; ./parboil clean bfs cuda
	$(SETENV) cd Parboil; ./parboil clean histo cuda
	$(SETENV) cd Parboil; ./parboil clean lbm cuda
	$(SETENV) cd Parboil; ./parboil clean mri-gridding cuda
	$(SETENV) cd Parboil; ./parboil clean mri-q cuda
	$(SETENV) cd Parboil; ./parboil clean sad cuda
	$(SETENV) cd Parboil; ./parboil clean sgemm cuda
	$(SETENV) cd Parboil; ./parboil clean spmv cuda
	$(SETENV) cd Parboil; ./parboil clean stencil cuda
	$(SETENV) cd Parboil; ./parboil clean tpacf cuda

clean_lonestar:
	$(setenv) make $(make_args) noinline=$(noinline) -C lonestargpu-2.0 clean

# clean_ispass-2009:
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/AES
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/BFS
# #	cd CP; export Parboil_ROOT=`pwd`; cd common/src; make $(MAKE_ARGS); cd -; ./Parboil compile cp cuda_short; cp benchmarks/cp/build/cuda_short/cp $(BINDIR)/CP 
# #	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG/3rdParty/ParMetis
# #	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/LIB
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/LPS
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/MUM
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/NN
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/NQU
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/RAY
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/STO
# 	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C ispass-2009/WP

clean_rodinia:
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/backprop 
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/bfs 
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/cfd
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hotspot 
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/kmeans
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/needle
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/streamingcluster
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/mummergpu
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/b+tree/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/dwt2d/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/heartwall/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/huffman/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hybridsort/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/myocyte/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/nn/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/particlefilter/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/particlefilter/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/pathfinder/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/lavaMD/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/lud/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/leukocyte/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/hotspot3D/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/gaussian
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C rodinia-3.1/srad/

# clean_rodinia_2.0-ft:
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/backprop
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/bfs
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/heartwall
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/hotspot
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/kmeans
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/lud
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/nn
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/nw
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/pathfinder
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/srad
# 	$(SETENV) make $(MAKE_ARGS) clean -C rodinia-3.1/2.0-ft/streamcluster

# clean_dragon-naive: 
# 	$(SETENV) rm -f /dragon_li/bin

# clean_dragon-cdp: 
# 	$(SETENV) rm -f /dragon_li/cdp_bin

# clean_pannotia: 
# 	$(SETENV) make $(MAKE_ARGS) clean -C pannotia/bc
# 	$(SETENV) export VARIANT="MAX"; make $(MAKE_ARGS) clean -C pannotia/color
# 	$(SETENV) export VARIANT="MAXMIN"; make $(MAKE_ARGS) clean -C pannotia/color
# 	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C pannotia/fw
# 	$(SETENV) export VARIANT="BLOCK"; make $(MAKE_ARGS) clean -C pannotia/fw
# 	$(SETENV) make $(MAKE_ARGS)  -C pannotia/mis
# 	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C pannotia/pagerank
# 	$(SETENV) export VARIANT="SPMV"; make $(MAKE_ARGS) clean -C pannotia/pagerank
# 	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C pannotia/sssp
# 	$(SETENV) export VARIANT="ELL"; make $(MAKE_ARGS) clean -C pannotia/sssp

# clean_proxy-apps:
# 	$(SETENV) make $(MAKE_ARGS) clean -C proxy-apps-doe/lulesh
# 	$(SETENV) make $(MAKE_ARGS) clean -C proxy-apps-doe/minife_matvec_ell
# 	$(SETENV) make $(MAKE_ARGS) clean -C proxy-apps-doe/xsbench
# 	chmod +x proxy-apps-doe/cns/compile.bash
# 	(cd proxy-apps-doe/cns/ ; ./compile.bash -c)
# 	chmod +x proxy-apps-doe/comd/clean.sh
# 	( cd proxy-apps-doe/comd ; ./clean.sh ) 

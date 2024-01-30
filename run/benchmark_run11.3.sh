#!/bin/bash

if [ ! -n "$1" ]; then
    echo "usage: $0 BENCH_NAME"
    exit 0
fi

echo "input new DIRNAME for simulation results"
read OUTDIR
mkdir -p $OUTDIR


if [ ! -d "$OUTDIR" ]; then
        echo "No directory"
        exit 0
fi


#IN=$1 #bench list

BENCH=$1

RODINIA_DATA=~/workspace/gpu_bench_cuda_11.3/data_dirs/cuda/rodinia/3.1
PARBOIL_DATA=~/workspace/gpu_bench_cuda_11.3/data_dirs/parboil/datasets
ISPASS_DATA=~/workspace/gpu_bench_cuda_11.3/data_dirs/cuda/ispass-2009/ispass-2009-
LONESTAR_DATA=~/workspace/gpu_bench_cuda_11.3/data_dirs/cuda/lonestargpu-2.0/inputs
APPBIN=$BINDIR
RODINIA_BIN=${APPBIN}/rodinia-3.1
PARBOIL_BIN=${APPBIN}/parboil
POLYBENCH_BIN=${APPBIN}/polybench
ISPASS_BIN=${APPBIN}/ispass-2009
LONESTAR_BIN=${APPBIN}/lonestargpu-2.0

OUT_DIR=~/workspace/gpu_bench_cuda_11.3/run/${OUTDIR}

OUT=""
BIN=""
DSET=""
IDATA=""
ODATA=""
PAR=""

#while read line
#do
#	BENCH="$line"

#	if [ -n "$BENCH" ]; then

case "$BENCH" in
#rodinia
bpr)
	BIN="${RODINIA_BIN}/backprop"
	IDATA="65536"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/bpr.txt"
	;;
bfs)
	BIN="${RODINIA_BIN}/bfs"
	IDATA="${RODINIA_DATA}/bfs-rodinia-3.1/data/graph1MW_6.txt"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/bfs.txt"
	;;

btr)
	BIN="${RODINIA_BIN}/b+tree"
	IDATA="file ${RODINIA_DATA}/b+tree-rodinia-3.1/data/mil.txt command ${RODINIA_DATA}/b+tree/data/command.txt"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/btr.txt"
	;;
dwt)
	BIN="${RODINIA_BIN}/dwt2d"
	IDATA="${RODINIA_DATA}/dwt2d-rodinia-3.1/data/rgb.bmp"
	ODATA="-d 1024x1024 -f -5 -l 3"
	PAR=""
	OUT="${OUT_DIR}/dwt.txt"
	;;
gaf)
	BIN="${RODINIA_BIN}/gaussian"
	IDATA="-f ${RODINIA_DATA}/gaussian-rodinia-3.1/data/matrix1024.txt"
	ODATA=""
	PAR=""
	OUT="${OUT_DIR}/gaf.txt"
	;;
# ga0)
# 	BIN="${RODINIA_BIN}/gaussian"
# 	IDATA="-f ${RODINIA_DATA}/gaussian/data/matrix208.txt"
# 	ODATA=""
#         PAR=""
#         OUT="${OUT_DIR}/ga0.txt"
# 	;;
# ga1)
# 	BIN="${RODINIA_BIN}/gaussian"
# 	IDATA="-f ${RODINIA_DATA}/gaussian/data/matrix16.txt"
# 	ODATA=""
#         PAR=""
#         OUT="${OUT_DIR}/ga1.txt"
# 	;;
# ga2)
# 	BIN="${RODINIA_BIN}/gaussian"
# 	IDATA="-f ${RODINIA_DATA}/gaussian/data/matrix4.txt"
# 	ODATA=""
#         PAR=""
#         OUT="${OUT_DIR}/ga2.txt"
# 	;;
# ga3)
# 	BIN="${RODINIA_BIN}/gaussian"
# 	IDATA="-f ${RODINIA_DATA}/gaussian/data/matrix3.txt"
# 	ODATA=""
#         PAR=""
#         OUT="${OUT_DIR}/ga3.txt"
# 	;;
# gas)
# 	BIN="${RODINIA_BIN}/gaussian"
# 	IDATA="-s 16"
# 	ODATA=""
#         PAR=""
#         OUT="${OUT_DIR}/gas.txt"
# 	;;
htw)
	BIN="${RODINIA_BIN}/heartwall"
	IDATA="${RODINIA_DATA}/heartwall-rodinia-3.1/data/test.avi"
	ODATA="20"
	PAR=""
	OUT="${OUT_DIR}/htw.txt"
	#PAR="5"
	;;
hsp)
	BIN="${RODINIA_BIN}/hotspot"
	IDATA="512 2 2 ${RODINIA_DATA}/hotspot-rodinia-3.1/data/temp_512 ${RODINIA_DATA}/hotspot-rodinia-3.1/data/power_512"
	ODATA="output.out"
	PAR=""
	OUT="${OUT_DIR}/hsp.txt"
	;;
# hs0)
# 	BIN="${RODINIA_BIN}/hotspot"
# 	IDATA="64 2 2 ${RODINIA_DATA}/hotspot/data/temp_64 ${RODINIA_DATA}/hotspot/power_64"
# 	ODATA="output.out"
# 	PAR=""
#         OUT="${OUT_DIR}/hs0.txt"
# 	;;
# hs1)
# 	BIN="${RODINIA_BIN}/hotspot"
# 	IDATA="1024 2 2 ${RODINIA_DATA}/hotspot/data/temp_1024 ${RODINIA_DATA}/hotspot/power_1024"
# 	ODATA="output.out"
# 	PAR=""
#         OUT="${OUT_DIR}/hs1.txt"
# 	;;
hsr)
	BIN="${RODINIA_BIN}/hybridsort"
	IDATA="${RODINIA_DATA}/hybridsort-rodinia-3.1/data/500000.txt"
	#PAR="r"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/hsr.txt"
	;;
kmn)
	BIN="${RODINIA_BIN}/kmeans"
	IDATA="-o -i ${RODINIA_DATA}/kmeans-rodinia-3.1/data/kdd_cup"
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/kmn.txt"	
	;;
lud)
	BIN="${RODINIA_BIN}/lud"
	IDATA="-i ${RODINIA_DATA}/lud-rodinia-3.1/data/2048.dat" 
	#IDATA="-s 256 -v"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/lud.txt"
	;;
mmg)
	BIN="${RODINIA_BIN}/mummergpu"
	IDATA="${RODINIA_DATA}/mummergpu-rodinia-3.1/data/NC_003997.fna ${RODINIA_DATA}/mummergpu-rodinia-3.1/data/NC_003997_q100bp.fna"
	ODATA="> NC_00399.out"
	PAR=""
	OUT="${OUT_DIR}/mmg.txt"
	;;
myo)
	BIN="${RODINIA_BIN}/myocyte"
	IDATA="100 1 0"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/myo.txt"
	;;
nn)
	#echo "${RODINIA_DATA}/nn/cane4_0.db" >> filelist_4
	#echo "${RODINIA_DATA}/nn/cane4_1.db" >> filelist_4
	#echo "${RODINIA_DATA}/nn/cane4_2.db" >> filelist_4
	#echo "${RODINIA_DATA}/nn/cane4_3.db" >> filelist_4
	BIN="${RODINIA_BIN}/nn"
	IDATA="${RODINIA_DATA}/nn-rodinia-3.1/data/filelist_4"
	ODATA="-r 5 -lat 30 -lng 90"
	PAR=""
	OUT="${OUT_DIR}/nn.txt"
	;;
nw)
	BIN="${RODINIA_BIN}/needle"
	IDATA="2048 10"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/nw.txt"
	;;
pfn)
	BIN="${RODINIA_BIN}/particlefilter_naive"
	IDATA="-x 128 -y 128 -z 10 -np 1000"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/pfn.txt"
	;;
pff)
	BIN="${RODINIA_BIN}/particlefilter_float"
	IDATA="-x 128 -y 128 -z 10 -np 1000"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/pff.txt"
	;;
pth)
	BIN="${RODINIA_BIN}/pathfinder"
	IDATA="100000 100 20"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/pth.txt"
	;;
sr1)
	BIN="${RODINIA_BIN}/srad1"
	IDATA="100 0.5 502 458"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/sr1.txt"
	;;
sr2)
	BIN="${RODINIA_BIN}/srad2"
	IDATA="2048 2048 0 127 0 127 0.5 2"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/sr2.txt"
	;;
scg)
	BIN="${RODINIA_BIN}/streamingcluster"
	IDATA="10 20 256 65536 65536 1000 none"
	ODATA="output.txt"
	PAR="1"
	OUT="${OUT_DIR}/sc.txt"
	;;
lav)
	BIN="${RODINIA_BIN}/lavaMD"
	IDATA="-boxes1d 10"
	ODATA=""
        PAR=""
	OUT="${OUT_DIR}/lav.txt"
	;;
huf)
        BIN="${RODINIA_BIN}/huffman"
        IDATA="${RODINIA_DATA}/huffman-rodinia-3.1/data/test1024_H2.206587175259.in"
        ODATA=""
	PAR=""
	OUT="${OUT_DIR}/huf.txt"
	;;
h3d)
        BIN="${RODINIA_BIN}/hotspot3D"
        IDATA="512 8 100"
	ODATA="${RODINIA_DATA}/hotspot3D-rodinia-3.1/data/power_512x8 ${RODINIA_DATA}/hotspot3D-rodinia-3.1/data/temp_512x8"
	PAR="output.out"
        OUT="${OUT_DIR}/h3d.txt"
        ;;
lek)
        BIN="${RODINIA_BIN}/leukocyte"
        IDATA="${RODINIA_DATA}/leukocyte-rodinia-3.1/data/testfile.avi"
        ODATA="5"
	PAR=""
	OUT="${OUT_DIR}/lek.txt"
        ;;
e3)	BIN="${RODINIA_BIN}/euler3d"
	IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.097K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3.txt"
	;;
e3f)    BIN="${RODINIA_BIN}/euler3d"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1data/fvcorr.domn.193K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3f.txt"
        ;;
e3m)    BIN="${RODINIA_BIN}/euler3d"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/missile.domn.0.2M"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3m.txt"
	;;

e3d)    BIN="${RODINIA_BIN}/euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.097K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3d.txt"
        ;;
e3df)   BIN="${RODINIA_BIN}/euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.193K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3df.txt"
        ;;
e3dm)   BIN="${RODINIA_BIN}/euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/missile.domn.0.2M"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/e3dm.txt"
        ;;

pe3)    BIN="${RODINIA_BIN}/pre_euler3d"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.097K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3.txt"
        ;;
pe3f)   BIN="${RODINIA_BIN}/pre_euler3d"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.193K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3f.txt"
        ;;
pe3m)   BIN="${RODINIA_BIN}/pre_euler3d"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/missile.domn.0.2M"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3m.txt"
        ;;

pe3d)   BIN="${RODINIA_BIN}/pre_euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.097K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3d.txt"
        ;;
pe3df)  BIN="${RODINIA_BIN}/pre_euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/fvcorr.domn.193K"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3df.txt"
        ;;
pe3dm)  BIN="${RODINIA_BIN}/pre_euler3d_double"
        IDATA="${RODINIA_DATA}/cfd-rodinia-3.1/data/missile.domn.0.2M"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pe3dm.txt"
        ;;
#rodinia end
#parboil
bf1)
        BIN=${PARBOIL_BIN}/bfs
        DSET=${PARBOIL_DATA}/bfs/1M/input
        IDATA="-i ${DSET}/graph_input.dat"
        ODATA="-o bfs.out"
	PAR=""
	OUT="${OUT_DIR}/bf1.txt"
#       ${PARBOIL_DIR} run bfs cuda 1M
        ;;
bfn)
        BIN=${PARBOIL_BIN}/bfs
        DSET=${PARBOIL_DATA}/bfs/NY/input
        IDATA="-i ${DSET}/graph_input.dat"
        ODATA="-o bfs.out"
	PAR=""
	OUT="${OUT_DIR}/bfn.txt"
#       ${PARBOIL_DIR} run bfs cuda NY
        ;;
bff)
        BIN=${PARBOIL_BIN}/bfs
        DSET=${PARBOIL_DATA}/bfs/SF/input
        IDATA="-i ${DSET}/graph_input.dat"
        ODATA="-o bfs.out"
	PAR=""
	OUT="${OUT_DIR}/bff.txt"
#       ${PARBOIL_DIR} run bfs cuda SF
        ;;
bfu)
        BIN=${PARBOIL_BIN}/bfs
        DSET=${PARBOIL_DATA}/bfs/UT/input
        IDATA="-i ${DSET}/graph_input.dat"
        ODATA="-o bfs.out"
	PAR=""
	OUT="${OUT_DIR}/bfu.txt"
#       ${PARBOIL_DIR} run bfs cuda UT
        ;;
cut)
        BIN=${PARBOIL_BIN}/cutcp
        DSET=${PARBOIL_DATA}/cutcp/small/input
        IDATA="-i ${DSET}/watbox.sl40.pqr"
        ODATA="-o lattice.dat"
	PAR=""
	OUT="${OUT_DIR}/cut.txt"
        ;;
his)
        BIN=${PARBOIL_BIN}/histo
        DSET=${PARBOIL_DATA}/histo/default/input
        IDATA="-i ${DSET}/img.bin"
        ODATA="-o ref.bmp"
        PAR="-- 20 4"
	OUT="${OUT_DIR}/his.txt"
        ;;
lbm)
        BIN=${PARBOIL_BIN}/lbm
        DSET=${PARBOIL_DATA}/lbm/short/input
        IDATA="-i ${DSET}/120_120_150_ldc.of"
        ODATA="-o reference.dat"
        PAR="-- 100"
	OUT="${OUT_DIR}/lbm.txt"
        #PAR="-- 3000"
        ;;
mrg)
        BIN=${PARBOIL_BIN}/mri-gridding
        DSET=${PARBOIL_DATA}/mri-gridding/small/input
        IDATA="-i ${DSET}/small.uks"
        ODATA="-o output.txt"
        PAR="-- 32 0"
	OUT="${OUT_DIR}/mrg.txt"
        ;;
mrq)
        BIN=${PARBOIL_BIN}/mri-q
        DSET=${PARBOIL_DATA}/mri-q/small/input
        IDATA="-i ${DSET}/32_32_32_dataset.bin"
        ODATA="-o 32_32_32_dataset.out"
	PAR=""
	OUT="${OUT_DIR}/mrq.txt"
        ;;
sad)
        BIN=${PARBOIL_BIN}/sad
        DSET=${PARBOIL_DATA}/sad/default/input
        IDATA="-i ${DSET}/reference.bin,${DSET}/frame.bin"
        ODATA="-o out.bin"
	PAR=""
	OUT="${OUT_DIR}/sad.txt"
        ;;
sge)
        BIN=${PARBOIL_BIN}/sgemm
        DSET=${PARBOIL_DATA}/sgemm/medium/input
        IDATA="-i ${DSET}/matrix1.txt,${DSET}/matrix2t.txt,${DSET}/matrix2t.txt"
        ODATA="-o matrix3.txt"
	PAR=""
	OUT="${OUT_DIR}/sge.txt"
        ;;
sgs)
        BIN=${PARBOIL_BIN}/sgemm
        DSET=${PARBOIL_DATA}/sgemm/small/input
        IDATA="-i ${DSET}/matrix1.txt,${DSET}/matrix2.txt,${DSET}/matrix2t.txt"
        ODATA="-o matrix3.txt"
        PAR=""
        OUT="${OUT_DIR}/sgs.txt"
	;;
spm)
        BIN=${PARBOIL_BIN}/spmv
        DSET=${PARBOIL_DATA}/spmv/medium/input
        IDATA="-i ${DSET}/bcsstk18.mtx,${DSET}/vector.bin"
        ODATA="-o bcsstk18.out"
	PAR=""
	OUT="${OUT_DIR}/spm.txt"
        ;;
sps)
        BIN=${PARBOIL_BIN}/spmv
        DSET=${PARBOIL_DATA}/spmv/small/input
        IDATA="-i ${DSET}/1138_bus.mtx,${DSET}/vector.bin"
        ODATA="-o 1138_bus.mtx.out"
        PAR=""
        OUT="${OUT_DIR}/sps.txt"
        ;;

sp1)
        BIN=${PARBOIL_BIN}/spmv
        DSET=${PARBOIL_DATA}/spmv/large/input
        IDATA="-i ${DSET}/Dubcova3.mtx.bin,${DSET}/vector.bin"
        ODATA="-o Dubcova3.out"
	PAR=""
	OUT="${OUT_DIR}/sp1.txt"
        ;;
ste)
        BIN=${PARBOIL_BIN}/stencil
        DSET=${PARBOIL_DATA}/stencil/default/input
        IDATA="-i ${DSET}/512x512x64x100.bin"
        ODATA="-o 512x512x64.out"
        PAR="-- 512 512 64 100"
	OUT="${OUT_DIR}/ste.txt"
        ;;
tpc)
        BIN=${PARBOIL_BIN}/tpacf
        DSET=${PARBOIL_DATA}/tpacf/medium/input
	IDATA="-i ${DSET}/Datapnts.1,${DSET}/Randompnts.1,${DSET}/Randompnts.2,${DSET}/Randompnts.3,${DSET}/Randompnts.4,${DSET}/Randompnts.5,${DSET}/Randompnts.6,${DSET}/Randompnts.7,${DSET}/Randompnts.8,${DSET}/Randompnts.9,${DSET}/Randompnts.10,${DSET}/Randompnts.11,${DSET}/Randompnts.12,${DSET}/Randompnts.13,${DSET}/Randompnts.14,${DSET}/Randompnts.15,${DSET}/Randompnts.16,${DSET}/Randompnts.17,${DSET}/Randompnts.18,${DSET}/Randompnts.19,${DSET}/Randompnts.20,${DSET}/Randompnts.21,${DSET}/Randompnts.22,${DSET}/Randompnts.23,${DSET}/Randompnts.24,${DSET}/Randompnts.25,${DSET}/Randompnts.26,${DSET}/Randompnts.27,${DSET}/Randompnts.28,${DSET}/Randompnts.29,${DSET}/Randompnts.30,${DSET}/Randompnts.31,${DSET}/Randompnts.32,${DSET}/Randompnts.33,${DSET}/Randompnts.34,${DSET}/Randompnts.35,${DSET}/Randompnts.36,${DSET}/Randompnts.37,${DSET}/Randompnts.38,${DSET}/Randompnts.39,${DSET}/Randompnts.40,${DSET}/Randompnts.41,${DSET}/Randompnts.42,${DSET}/Randompnts.43,${DSET}/Randompnts.44,${DSET}/Randompnts.45,${DSET}/Randompnts.46,${DSET}/Randompnts.47,${DSET}/Randompnts.48,${DSET}/Randompnts.49,${DSET}/Randompnts.50,${DSET}/Randompnts.51,${DSET}/Randompnts.52,${DSET}/Randompnts.53,${DSET}/Randompnts.54,${DSET}/Randompnts.55,${DSET}/Randompnts.56,${DSET}/Randompnts.57,${DSET}/Randompnts.58,${DSET}/Randompnts.59,${DSET}/Randompnts.60,${DSET}/Randompnts.61,${DSET}/Randompnts.62,${DSET}/Randompnts.63,${DSET}/Randompnts.64,${DSET}/Randompnts.65,${DSET}/Randompnts.66,${DSET}/Randompnts.67,${DSET}/Randompnts.68,${DSET}/Randompnts.69,${DSET}/Randompnts.70,${DSET}/Randompnts.71,${DSET}/Randompnts.72,${DSET}/Randompnts.73,${DSET}/Randompnts.74,${DSET}/Randompnts.75,${DSET}/Randompnts.76,${DSET}/Randompnts.77,${DSET}/Randompnts.78,${DSET}/Randompnts.79,${DSET}/Randompnts.80,${DSET}/Randompnts.81,${DSET}/Randompnts.82,${DSET}/Randompnts.83,${DSET}/Randompnts.84,${DSET}/Randompnts.85,${DSET}/Randompnts.86,${DSET}/Randompnts.87,${DSET}/Randompnts.88,${DSET}/Randompnts.89,${DSET}/Randompnts.90,${DSET}/Randompnts.91,${DSET}/Randompnts.92,${DSET}/Randompnts.93,${DSET}/Randompnts.94,${DSET}/Randompnts.95,${DSET}/Randompnts.96,${DSET}/Randompnts.97,${DSET}/Randompnts.98,${DSET}/Randompnts.99,${DSET}/Randompnts.100"
        ODATA="-o tpacf.out"
        PAR="-- -n 100 -p 4096"
	OUT="${OUT_DIR}/tpc.txt"
        ;;
#parboil end
#polybench
cor)
	BIN=${POLYBENCH_BIN}/correlation
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/corr.txt
	;;
cov)
	BIN=${POLYBENCH_BIN}/covariance
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/cov.txt
	;;
2mm)
	BIN=${POLYBENCH_BIN}/2mm
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/2mm.txt
	;;
3mm)
	BIN=${POLYBENCH_BIN}/3mm
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/3mm.txt
	;;
atx)
	BIN=${POLYBENCH_BIN}/atax
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/atx.txt
	;;
bic)
	BIN=${POLYBENCH_BIN}/bicg
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/bic.txt
	;;
dit)
	BIN=${POLYBENCH_BIN}/doitgen
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/dit.txt
	;;
gmm)
	BIN=${POLYBENCH_BIN}/gemm
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/gmm.txt
	;;
gmv)
	BIN=${POLYBENCH_BIN}/gemver
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/gmv.txt
	;;
gsm)
	BIN=${POLYBENCH_BIN}/gesummv
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/gsm.txt
	;;
mvt)
	BIN=${POLYBENCH_BIN}/mvt
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/mvt.txt
	;;
syr)
	BIN=${POLYBENCH_BIN}/syrk
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/syr.txt
	;;
sy2)
	BIN=${POLYBENCH_BIN}/syr2k
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/sy2.txt
	;;
grm)
	BIN=${POLYBENCH_BIN}/gramschmidt
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/grm.txt
	;;
lu)
	BIN=${POLYBENCH_BIN}/lu
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/lu.txt
	;;
adi)
	BIN=${POLYBENCH_BIN}/adi
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/adi.txt
	;;
cv2)
	BIN=${POLYBENCH_BIN}/2DConvolution
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/cv2.txt
	;;
cv3)
	BIN=${POLYBENCH_BIN}/3DConvolution
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/cv3.txt
	;;
fdt)
	BIN=${POLYBENCH_BIN}/fdtd2d
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/fdt.txt
	;;
jc1)
	BIN=${POLYBENCH_BIN}/jacobi1D
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/jc1.txt
	;;
jc2)
	BIN=${POLYBENCH_BIN}/jacobi2D
        IDATA=""
        ODATA=""
        PAR=""
	OUT=${OUT_DIR}/jc2.txt
	;;
#polybench end


#lonestargpu-2.0
bfa)
        BIN="${LONESTAR_BIN}/lonestar-bfs-atomic"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bfa.txt"
        ;;
bfl)
        BIN="${LONESTAR_BIN}/lonestar-bfs"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bfl.txt"
        ;;
bfw)
        BIN="${LONESTAR_BIN}/lonestar-bfs-wlw"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bfw.txt"
        ;;
bfwa)
        BIN="${LONESTAR_BIN}/lonestar-bfs-wla"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bfwa.txt"
        ;;
	
bfc)
        BIN="${LONESTAR_BIN}/lonestar-bfs-wlc"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bfc.txt"
        ;;
ssp)
        BIN="${LONESTAR_BIN}/lonestar-sssp"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/ssp.txt"
        ;;
ssc)
        BIN="${LONESTAR_BIN}/lonestar-sssp-wlc"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/ssc.txt"
        ;;
ssn)
        BIN="${LONESTAR_BIN}/lonestar-sssp-wln"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/ssn.txt"
        ;;
bh)
        BIN="${LONESTAR_BIN}/lonestar-bh"
        IDATA="300000 10 0"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bh.txt"
        ;;
dmr)
        BIN="${LONESTAR_BIN}/lonestar-dmr"
        IDATA="${LONESTAR_DATA}/r5M.ele 20"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/dmr.txt"
        ;;
mst)
        BIN="${LONESTAR_BIN}/lonestar-mst"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/mst.txt"
        ;;
nsp)
        BIN="${LONESTAR_BIN}/lonestar-nsp"
        IDATA="${LONESTAR_DATA}/USA-road-d.USA.gr"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/nsp.txt"
        ;;
pta)
        BIN="${LONESTAR_BIN}/lonestar-pta"
        IDATA="${LONESTAR_DATA}/tshark_correct_soln_001.txt"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/pta.txt"
        ;;
#lonestargpu-2.0 end

#ispass-2009
aes)
        BIN="${ISPASS_BIN}/ispass-2009-AES"
        IDATA="e 128"
        ODATA="${ISPASS_DATA}AES/data/output.bmp"
        PAR="${ISPASS_DATA}AES/data/key128.txt"
        OUT="${OUT_DIR}/aes.txt"
        ;;
ibf)
        BIN="${ISPASS_BIN}/ispass-2009-BFS"
        IDATA="${ISPASS_DATA}BFS/data/graph65536.txt"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/ibf.txt"
        ;;
lib)
        BIN="${ISPASS_BIN}/ispass-2009-LIB"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/lib.txt"
        ;;
lps)
        BIN="${ISPASS_BIN}/ispass-2009-LPS"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/lps.txt"
        ;;
mum)
        BIN="${ISPASS_BIN}/ispass-2009-MUM"
        IDATA="${ISPASS_DATA}MUM/data/NC_003997.20k.fna"
        ODATA="${ISPASS_DATA}MUM/data/NC_003997_q25bp.50k.fna"
        PAR=""
        OUT="${OUT_DIR}/mum.txt"
        ;;
inn)
        BIN="${ISPASS_BIN}/ispass-2009-NN"
        IDATA="28"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/inn.txt"
        ;;
nqu)
        BIN="${ISPASS_BIN}/ispass-2009-NQU"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/nqu.txt"
        ;;
ray)
        BIN="${ISPASS_BIN}/ispass-2009-RAY"
        IDATA="256 256"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/ray.txt"
        ;;
sto)
        BIN="${ISPASS_BIN}/ispass-2009-STO"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/sto.txt"
        ;;
#sdk
alt)
	BIN="${SDK_BIN}/alignedTypes"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${SDK_DIR}/alt.txt"
	;;
asy)
	BIN="${SDK_BIN}/asyncAPI"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="asy.txt"
        ;;
bin)
	BIN="${SDK_BIN}/binomialOptions"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="bin.txt"
        ;;
blk)
	BIN="${SDK_BIN}/BlackScholes"
        IDATA=""
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/blk.txt"
	;;
#box)
#	${SDK_BIN}/boxFilter ${SDK_DIR}/boxFilter
#	;;
cnv)
	BIN="${SDK_BIN}/convolutionSeparable"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/cnv.txt"
	;;
cnt)
	BIN="${SDK_BIN}/convolutionTexture"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="cnt.txt"
        ;;
dct)
	BIN="${SDK_BIN}/dct8x8"
	IDATA="${SDK_DIR}/dct8x8/data/barbara.ppm"
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/dct.txt"
        ;;
dxt)
	cp ${SDK_DIR}/dxtc/data/* .
	BIN="${SDK_BIN}/dxtc >> gpugj.rpt"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/dxt.txt"
        ;;
eig)
	BIN="${SDK_BIN}/eigenvalues"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/eig.txt"
        ;;
hst)
	BIN="${SDK_BIN}/histogram"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/hst.txt"
	;;
mgs)
	BIN="${SDK_BIN}/mergeSort"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/mgs.txt"
	;;
mt)
	cp ${SDK_DIR}/MersenneTwister/data/* .
	BIN="${SDK_BIN}/MersenneTwister"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/mt.txt"
        ;;
mca)
	BIN="${SDK_BIN}/MonteCarlo"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/mca.txt"
        ;;
qsr)
	BIN="${SDK_BIN}/quasirandomGenerator"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/qsr.txt"
        ;;
red)
	BIN="${SDK_BIN}/reduction"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/red.txt"
        ;;
spd)
	BIN="${SDK_BIN}/scalarProd"
        IDATA=""
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/spd.txt"
	;;
scn)
	BIN="${SDK_BIN}/scan"
        IDATA=""
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/scn.txt"
	;;
sao)
	BIN="${SDK_BIN}/SingleAsianOptionP"
	IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/sao.txt"
        ;;
sbq)
	BIN="${SDK_BIN}/SobolQRNG"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/sbq.txt"
	;;
snt)
	BIN="${SDK_BIN}/sortingNetworks"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/snt.txt"
	;;
tfr)
	BIN="${SDK_BIN}/threadFenceReduction"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/tfr.txt"
	;;
txp)
	BIN="${SDK_BIN}/transpose"
        IDATA=""
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/txp.txt"
	;;
vad)
	BIN="${SDK_BIN}/vectorAdd"
        IDATA=""
        ODATA=""
        PAR=""
	OUT="${OUT_DIR}/vad.txt"
	;;
wal)
	BIN="${SDK_BIN}/fastWalshTransform"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/val.txt"
	;;
fd3)
        BIN="${SDK_BIN}/FDTD3d"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/fd3.txt"
        ;;
mxm)
        BIN="${SDK_BIN}/matrixMul"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/mxm.txt"
        ;;
smp)
        BIN="${SDK_BIN}/StreamPriorities"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/smp.txt"
        ;;
bop)
        BIN="${SDK_BIN}/binomialOptions"
        IDATA=""
        ODATA=""
        PAR=""
        OUT="${OUT_DIR}/bop.txt"
        ;;
*)
	echo "Invalid benchmark name!!!"
	;;
esac

echo "${BIN} ${IDATA} ${ODATA} ${PAR}"
${BIN} ${IDATA} ${ODATA} ${PAR} >> ${OUT}

#	fi
#done < ${IN}

rm -r _*
rm -r *ptx
rm -r *ptxas



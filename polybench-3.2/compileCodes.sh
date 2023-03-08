#!/bin/bash

# set PATH and LD_LIBRARY_PATH for CUDA/OpenCL installation (may need to be adjusted)
#export PATH=$PATH:/usr/local/cuda-10.1/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib:/usr/local/cuda-10.1/lib64

if [ -z "$1" ]; then
    echo "usage : $0 new_dir"
    exit 0
fi

outdir="$1"
mkdir ${outdir}
bin=""
for currDir in *
do
    if [ -d $currDir ] && [ $currDir != $outdir ] && [ $currDir != "utilities" ]; then
		echo $currDir
		cd $currDir
		make clean
		make
		bin="$currDir.exe"
		mv $bin ../$outdir
		cd ..
    fi
done

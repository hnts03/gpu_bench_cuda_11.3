#!/bin/sh
if [ -z "$1" ]; then
    echo "usage : $0 new_dir"
    exit 0
fi

outdir="$1"
chmod u+x ./parboil
chmod u+x benchmarks/*/tools/compare-output
mkdir ${outdir}
for currDir in benchmarks/*
do
    if [ -d $currDir ]; then
	cd $currDir
	bench=${PWD##*/}
	echo "$bench"
	cd ../..
	./parboil clean $bench cuda
	./parboil compile $bench cuda
		#make $1 clean
		#make $1
		#cd ..
	cp $currDir/build/cuda_default/$bench ${outdir}

    fi
done

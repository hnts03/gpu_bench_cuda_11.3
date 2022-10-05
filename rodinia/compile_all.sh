#!/bin/sh
if [ -z "$1" ]; then
    echo "usage : $0 new_dir"
    exit 0
fi

outdir="$1"
#chmod u+x ./parboil
#chmod u+x benchmarks/*/tools/compare-output
mkdir ${outdir}
for currDir in *
do
    if [ -d $currDir ]; then
	case $currDir in
	particlefilter)
		echo "compile $currDir"
		cd $currDir
		make clean
		make
		#bench1="${PWD##*/}_float"
		bench1="${currDir}_float"
		bench2="${currDir}_naive"
		#bench2="${PWD##*/}_naive"
		cp $bench1 ../${outdir}
		cp $bench2 ../${outdir}
		cd ..
	;;
	srad)
		echo "compile $currDir"
		cd $currDir
		make clean
		bench1="${currDir}_V1"
		bench2="${currDir}_V2"
		cd $bench1
		make
		bin1="${currDir}1"
		cp $bin1 ../../${outdir}
		cd ../$bench2
		make
		bin2="${currDir}2"
		cp $bin2 ../../${outdir}
		cd ../../
	;;
	cfd)
		echo "compile $currDir"
		cd $currDir
		make clean
		make
		cfd1="euler3d"
		cfd2="euler3d_double"
		cfd3="pre_euler3d"
		cfd4="pre_euler3d_double"
		cp $cfd1 ../${outdir}
		cp $cfd2 ../${outdir}
		cp $cfd3 ../${outdir}
		cp $cfd4 ../${outdir}
		cd ..
	;;
	*)
		cd $currDir
		#bench=${PWD##*/}
		echo "compile $currDir"
		make clean
		make
		cp $currDir ../${outdir}
		cd ..
	;;
	esac

    fi
done

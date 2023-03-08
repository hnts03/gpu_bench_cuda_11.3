#!/bin/sh
if [ -z "$1" ]; then
    echo "usage : $0 new_dir"
    exit 0
fi

outdir="$1"
mkdir ${outdir}
make clean
rm -r bin/*
make
cd apps
for currDir in *
do
    if [ -d $currDir ] && [ $currDir != $outdir ] && [ $currDir != "scripts" ]; then
	case $currDir in
	bfs)
		echo "collect binary $currDir"
		bench1="${currDir}-atomic"
		#bench2="${currDir}-merrill"
		bench3="${currDir}-wlw"
		bench4="${currDir}-wla"
		bench5="${currDir}-wlc"
		cp $currDir/$currDir ../${outdir}
		cp $currDir/$bench1 ../${outdir}
		#cp $currDir/$bench2 ../${outdir}
		cp $currDir/$bench3 ../${outdir}
		cp $currDir/$bench4 ../${outdir}
		cp $currDir/$bench5 ../${outdir}
	;;
	sssp)
		echo "collect binary $currDir"
		bench1="${currDir}-wln"
		bench2="${currDir}-wlc"
		cp $currDir/$currDir ../${outdir}
		cp $currDir/$bench1 ../${outdir}
		cp $currDir/$bench2 ../${outdir}
	;;
	*)
		echo "collect binary $currDir"
		cp $currDir/$currDir ../${outdir}
	;;
	esac

    fi
done

cd ..


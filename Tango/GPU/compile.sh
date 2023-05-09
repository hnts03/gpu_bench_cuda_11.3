#!/bin/bash

for currDir in *
do
	if [ -d $currDir ]; then
		echo $currDir
		cd $currDir
		sh build.sh
		cd ..
	fi
done

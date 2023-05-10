
echo "install dependencies for compile benchmarks"
sudo apt-get update
echo "install boost libraty"
sudo apt-get install libboost-all-dev
echo "install cmake-3.15.1"
wget https://cmake.org/files/v3.15/cmake-3.15.1.tar.gz
tar -xvf cmake-3.15.1.tar.gz
rm -r cmake-3.15.1.tar.gz
cd cmake-3.15.1
./bootstrap && make && sudo make install
cd ..
echo "install cudnn-8.5.0"
if [ -e "cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz" ];
    tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.3/include
    sudo cp -P cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.3/lib64
    sudo chmod a+r /usr/local/cuda/include-11.3/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*
    rm -r cudnn-linux-x86_64-8.5.0.96_cuda11-archive
else
    echo "No cudnn library installer"
fi
echo "install libfmt"
sudo apt-get install libfmt-dev
echo "install LLVM"
#wget https://apt.llvm.org/llvm.sh
#chmod +x llvm.sh
sudo ./llvm.sh 10
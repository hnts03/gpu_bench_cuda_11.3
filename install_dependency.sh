
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
echo "install cudnn-8.2.0"
if [ -e "cudnn-11.3-linux-x64-v8.2.0.53.tgz" ]; then
    tar -xvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
    sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.3/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64
    sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*
    rm -r cuda
else
    echo "No cudnn library installer"
fi
echo "install libfmt"
sudo apt-get install libfmt-dev
echo "install LLVM"
#wget https://apt.llvm.org/llvm.sh
#chmod +x llvm.sh
sudo ./llvm.sh 10

echo "install dependencies for compile benchmarks"
sudo apt-get update
echo "install boost library"
sudo apt-get install -y libboost-all-dev
#echo "install cmake-3.15.1"
#wget https://cmake.org/files/v3.15/cmake-3.15.1.tar.gz
#tar -xvf cmake-3.15.1.tar.gz
#rm -r cmake-3.15.1.tar.gz
#cd cmake-3.15.1
#./bootstrap && make && sudo make install
#cd ..
#echo "install cudnn-8.2.0"
#if [ -e "cudnn-11.3-linux-x64-v8.2.0.53.tgz" ]; then
#    tar -xvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
#    sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.3/include
#    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64
#    sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*
#    rm -r cuda
#else
#    echo "No cudnn library installer"
#fi
echo "install libfmt"
sudo apt-get install -y libfmt-dev
echo "install LLVM"
#wget https://apt.llvm.org/llvm.sh
#chmod +x llvm.sh
sudo ./llvm.sh 10

# install pytorch and dependencies for GNNmark
# wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# chmod +x Anaconda3-2023.03-1-Linux-x86_64.sh
# ./Anaconda3-2023.03-1-Linux-x86_64.sh

# source ~/.bashrc
# conda create -n gnnmark python=3.7 anaconda
# conda activate gnnmark

# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html

# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# pip install torch_geometric    # 2.3.1

# pip install -r requirements.txt
# grapwritter, stgcn: ./download.sh

THIS README is still being udpated


System GNNMark has been tested on 
1. Ubuntu 16.04
2. GCC/G++ (5.4.0)
3. CUDA 10.2
4. PyTorch 1.7.0
5. DGL 0.5.2
6. PyTorch Geometric 1.6.1

You should be able to use other ubuntu and newer gcc versions without running into issues. These are just the specs of the system we used.


**CUDA requirements**
1. Currently GNNMark workloads have been tested with CUDA 10.2. To install CUDA 10.2 on your machine follow the instructions here(https://developer.nvidia.com/cuda-10.2-download-archive)


**Python Virtual Environment**
Using a virtual environment for python is highly recommended to avoid package conflicts in your current system

1. You will need to install conda first (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Next step is to set up a python virtual environment with python 3.7. This will help ensure your python environment on the local system remains clean and free of conflicts
3. Use the command conda create -n ENV_NAME python=3.7 anaconda
4. Now activate the virtual environment using the command “source activate ENV_NAME"

**Important**: All the next steps should be performed inside your created virtual environment to avoid python package conflicts. Also do not use “sudo” inside the virtual environment.

**Install PyTorch**
Run "conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch"

**Install DGL**
Run "conda install -c dglteam dgl-cuda10.2"

**Install PyTorch Geometric**
1. pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
2. pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
3. pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
4. pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
5. pip install torch-geometric


**Additional Requirements**
1. All other additional requirements that are specific to a particular application will be mentioned inside the respective application directory
2. The user can easily use the provided “requirements.txt”  file which is provided(if required) inside the application directory with the command “pip install -r requirements.txt” to install those requirements

**Multi-GPU requirements**
No additional software is required to run the programs with multiple GPUs. However, you do have to set an IP address and a port in the application. These two parameters are os.environ['MASTER_ADDR'] and os.environ['MASTER_PORT']. Here you have to set the IP of the root process such as 129.xx.xx.xx and a port such as 8888. The file for each application where you have to set this will be mentioned inside the individual README’s that are part of the application directory


| App        | Datasets Supported | MultiGPU Support |
| ------------- |:-------------:| -----:|
| DeepGCN     | ogbg-molhiv,ogbg-molbace,ogbg-molpcba | Yes  |
| GraphWriter | AGENDA      |   Yes |
| STGCN       | LA,PEMS-BAY      |    Yes |
| KGNN        | PROTEINS         | Yes  |
| ARGA        | Cora,CiteSeer,PubMed     |   No |
| TreeLSTM    | SST, SICK(In Progress)     |    Yes |
| PinSAGE     | MovieLens,NowPlaying   |    Yes |
| TGN(In Progress)         | Wikipedia,Reddit  |    Yes |
| GCRL(In Progress)         | MAgent  |    Yes |








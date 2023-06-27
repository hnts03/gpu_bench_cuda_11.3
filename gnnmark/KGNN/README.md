**Benchmark**
This benchmark runs the k Graph Neural Network(k-GNN) GNN for protein classification. The directory contains two training scripts 1-proteins.py where k=1 and 1-2-3-proteins.py which is a higher order k-GNN where k=3


**Supported Datasets**
The application supports the protein dataset currently.

**Datasets Download**
1. The dataset will be automatically downloaded by the training script


**Additional Requirements**
1. This benchmark does not need any additional packages. Hence no requirements.txt file is provided
2. You must run "python setup.py build" to build some of the library code required by the application




**Configuring**
1. Before running, you must add the following two parameters inside the main function in both 1-proteins.py and 1-2-3-proteins.py
    1. os.environ['MASTER_ADDR']
    2. os.environ['MASTER_PORT']


**Execution Example**
1. To run 1-proteins.py use the following command : python 1-proteins.py -n 1 -g 1 -nr 0 --epochs 32 --batch 32
2. To run 1-2-3-proteins.py, use the following command: python 1-2-3-proteins.py -n 1 -g 1 -nr 0 --epochs 32 --batch 32

**Multi-GPU Support**
1. KGNN supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python 1-proteins.py -n 1 -g 4 -nr 0 --epochs 32 --batch 32, python 1-2-3-proteins.py -n 1 -g 4 -nr 0 --epochs 32 --batch 32
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{morris2019weisfeiler,
  title={Weisfeiler and leman go neural: Higher-order graph neural networks},
  author={Morris, Christopher and Ritzert, Martin and Fey, Matthias and Hamilton, William L and Lenssen, Jan Eric and Rattan, Gaurav and Grohe, Martin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={4602--4609},
  year={2019}
}
```


```
@article{fey2019fast,
  title={Fast graph representation learning with PyTorch Geometric},
  author={Fey, Matthias and Lenssen, Jan Eric},
  journal={arXiv preprint arXiv:1903.02428},
  year={2019}
}
```

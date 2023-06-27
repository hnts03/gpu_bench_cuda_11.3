**Benchmark**
This benchmark runs the Spatio Temporal Graph Convolutional Networks(STGCN) GNN for traffic forecasting

**Supported Datasets**
The following datasets are supported:
1. LA:  Pass the argument string " --tsfilepath ./data/metr-la.h5  "
2. PEMS-BAY: Pass the argument string " --tsfilepath ./data/pems-bay.h5  --pems_bay true "



**Datasets Download**
1. The datasets can be downloaded and placed into the appropriate directory by running the script "./download.sh"


**Additional Requirements**
1. This benchmark does not need any additional packages. Hence no requirements.txt file is provided



**Configuring**
1. Before running, you must add the following two parameters inside the main function in train.py
    1. os.environ['MASTER_ADDR']
    2. os.environ['MASTER_PORT']


**Execution Example**
1. To run the benchmark on LA dataset use the following command : python main.py --tsfilepath ./data/metr-la.h5  -n 1 -g 1 -nr 0 --epochs 32 --batch_size 16

**Multi-GPU Support**
1. STGCN supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python main.py --tsfilepath ./data/metr-la.h5  -n 1 -g 4 -nr 0 --epochs 32 --batch_size 16
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{yu2018spatio,
    title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
    author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
    booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2018}
}
```


```
@article{wang2019deep,
  title={Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs.},
  author={Wang, Minjie and Yu, Lingfan and Zheng, Da and Gan, Quan and Gai, Yu and Ye, Zihao and Li, Mufei and Zhou, Jinjing and Huang, Qi and Ma, Chao and others},
  year={2019}
}
```

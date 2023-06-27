**Benchmark**
This benchmark runs the Tree LSTM GNN for Sentiment Classification 

**Supported Datasets**
Currently Tree LSTM supports and loads only one dataset. We are working on supporting additional datasets:
1. Stanford Sentiment TreeBank(SST)



**Datasets Download**
1. The datasets will be downloaded automatically by the training script


**Additional Requirements**
1. This benchmark does not need any additional packages. Hence no requirements.txt file is provided


**Configuring**
1. Before running, you must add the following two parameters inside the main function in train.py:
	1. os.environ['MASTER_ADDR']
	2. os.environ['MASTER_PORT']

**Execution Example**
1. To run the benchmark use the following command : python train.py -n 1 -g 1 -nr 0 --epochs 32 --batch-size 128

**Multi-GPU Support**
1. Tree LSTM supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python train.py -n 1 -g 4 -nr 0 --epochs 32 --batch-size 128
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{tai-etal-2015-improved,
    title = "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks",
    author = "Tai, Kai Sheng  and
      Socher, Richard  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul,
    year = "2015",
    address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P15-1150",
    doi = "10.3115/v1/P15-1150",
    pages = "1556--1566",
}
```

```
@article{wang2019deep,
  title={Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs.},
  author={Wang, Minjie and Yu, Lingfan and Zheng, Da and Gan, Quan and Gai, Yu and Ye, Zihao and Li, Mufei and Zhou, Jinjing and Huang, Qi and Ma, Chao and others},
  year={2019}
}
```

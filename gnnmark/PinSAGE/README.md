**Benchmark**
This benchmark runs the PinSAGE GNN used for recommendation purposes

**Supported Datasets**
The following datasets are supported:
1. MovieLens:  Pass the argument string " --dataset_path ./data/movielens.pkl  "
2. NowPlaying: Pass the argument string " --dataset_path ./data/nowplaying.pkl "



**Datasets Download**
1. The datasets can be downloaded and placed into the appropriate directory by running the script "./download.sh"


**Additional Requirements**
1. This benchmark needs additional packages
2. Run pip install -r requirements.txt " UPDATE THIS FILE"



**Configuring**
1. Before running, you must add the following two parameters inside the main function in model.py:
    1. os.environ['MASTER_ADDR']
    2. os.environ['MASTER_PORT']


**Execution Example**
1. To run the benchmark on Movielens dataset use the following command : python model.py --dataset_path ./data/movielens.pkl --num-epochs 10 --batch-size 256  -n 1 -g 1 -nr 0 

**Multi-GPU Support**
1. PinSAGE supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python model.py --dataset_path ./data/movielens.pkl --num-epochs 10 --batch-size 256   -n 1 -g 4 -nr 0 
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"
4. When using multi-GPU, ensure that the argument " --num-workers " is set to zero as otherwise Pytorch DDP will throw an error during execution.

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{ying2018graph,
  title={Graph convolutional neural networks for web-scale recommender systems},
  author={Ying, Rex and He, Ruining and Chen, Kaifeng and Eksombatchai, Pong and Hamilton, William L and Leskovec, Jure},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={974--983},
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

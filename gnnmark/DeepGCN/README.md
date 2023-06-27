**Benchmark**
This benchmark runs the Deep Graph Convolutional Networks(DeepGCN) GNN for protein molecule classification

**Supported Datasets**
The following datasets from the OGB datasets are supported:
1. ogbg-molhiv:  Pass the argument string " --dataset ogbg-molhiv  "
2. ogbg-molbace: Pass the argument string " --dataset ogbg-molbace "
3. ogbg-molpcba: Pass the argument string " --dataset ogbg-molpcba "



**Datasets Download**
1. The datasets will be automatically downloaded the first time when that dataset is passed as an argument to the dataset string


**Additional Requirements**
1. This benchmark has additional requirements.
2. Run "pip install -r requirements.txt"



**Configuring**
1. Before running, you must add the following two parameters inside the __init__ function in args.py
    1. os.environ['MASTER_ADDR']
    2. os.environ['MASTER_PORT']


**Execution Example**
1. To run the benchmark on "ogbg-molhiv"  dataset use the following command : python main.py --conv_encode_edge --num_layers 28  --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2  --epochs 1 --batch_size 256 --dataset ogbg-molhiv -n 1 -g 1 -nr 0

**Multi-GPU Support**
1. DeepGCN supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python main.py --conv_encode_edge --num_layers 28  --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2  --epochs 1 --batch_size 256 --dataset ogbg-molhiv -n 1 -g 4 -nr 0
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@InProceedings{li2019deepgcns,
    title={DeepGCNs: Can GCNs Go as Deep as CNNs?},
    author={Guohao Li and Matthias Müller and Ali Thabet and Bernard Ghanem},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```


```
@misc{li2019deepgcns_journal,
    title={DeepGCNs: Making GCNs Go as Deep as CNNs},
    author={Guohao Li and Matthias Müller and Guocheng Qian and Itzel C. Delgadillo and Abdulellah Abualshour and Ali Thabet and Bernard Ghanem},
    year={2019},
    eprint={1910.06849},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```
@misc{li2020deepergcn,
    title={DeeperGCN: All You Need to Train Deeper GCNs},
    author={Guohao Li and Chenxin Xiong and Ali Thabet and Bernard Ghanem},
    year={2020},
    eprint={2006.07739},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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

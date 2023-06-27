**Benchmark**
This benchmark runs the Adverserially Regularized Graph Autoencoder(ARGA) GNN used for node classification

**Supported Datasets**
These names can be passed to the "dataset" argument in the training script
1. Cora
2. CiteSeer
3. PubMed



**Datasets Download**
1. The datasets will be downloaded automatically by the training script


**Additional Requirements**
1. This benchmark does not need any additional packages. Hence no requirements.txt file is provided

**Execution Example**
1. To run the benchmark on the PubMed dataset use the following command : python argva_node_clustering.py --dataset PubMed

**Multi-GPU Support**
1. ARGA does not support Multi-GPU execution at the moment


**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{pan2018adversarially,
  title={Adversarially Regularized Graph Autoencoder for Graph Embedding.},
  author={Pan, Shirui and Hu, Ruiqi and Long, Guodong and Jiang, Jing and Yao, Lina and Zhang, Chengqi},
  booktitle={IJCAI},
  pages={2609--2615},
  year={2018}
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


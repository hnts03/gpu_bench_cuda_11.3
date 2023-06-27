**Benchmark**
This benchmark runs the GraphWriter(GW) GNN for text generation from knowledge graphs

**Supported Datasets**
GrapWriter currently supports only the default dataset which is the AGENDA dataset


**Datasets Download**
1. The datasets can be downloaded and placed into the appropriate directory by running the script "./download.sh"


**Additional Requirements**
1. This benchmark needs additional requirements
2. Run "pip install -r requirements.txt" to install them



**Configuring**
1. Before running, you must add the following two parameters inside the main function in train.py
    1. os.environ['MASTER_ADDR']
    2. os.environ['MASTER_PORT']


**Execution Example**
1. To run the benchmark, use the following command : python train.py -n 1 -g 1 -nr 0 --epoch 10  --batch_size 32

**Multi-GPU Support**
1. GW supports multi-GPU training. This can be controlled by the parameters "n,g,nr". "n" is the number of nodes. "g" is the number of GPUs on each node. "nr" is the rank of the master process
2. Example execution on 4 GPUs on one node is : python train.py  -n 1 -g 4 -nr 0 --epoch 10 --batch_size 32
3. Note the batch size parameter is for each GPU. So the effective batch size is "batch_size x g x n"

**Citations**
1. If you use this benchmark, please cite the following papers:

```
@inproceedings{koncel-kedziorski-etal-2019-text,
    title = "{T}ext {G}eneration from {K}nowledge {G}raphs with {G}raph {T}ransformers",
    author = "Koncel-Kedziorski, Rik  and
      Bekal, Dhanush  and
      Luan, Yi  and
      Lapata, Mirella  and
      Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1238",
    doi = "10.18653/v1/N19-1238",
    pages = "2284--2293",
    abstract = "Generating texts which express complex ideas spanning multiple sentences requires a structured representation of their content (document plan), but these representations are prohibitively expensive to manually produce. In this work, we address the problem of generating coherent multi-sentence texts from the output of an information extraction system, and in particular a knowledge graph. Graphical knowledge representations are ubiquitous in computing, but pose a significant challenge for text generation techniques due to their non-hierarchical nature, collapsing of long-distance dependencies, and structural variety. We introduce a novel graph transforming encoder which can leverage the relational structure of such knowledge graphs without imposing linearization or hierarchical constraints. Incorporated into an encoder-decoder setup, we provide an end-to-end trainable system for graph-to-text generation that we apply to the domain of scientific text. Automatic and human evaluations show that our technique produces more informative texts which exhibit better document structure than competitive encoder-decoder methods.",
}
```


```
@article{wang2019deep,
  title={Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs.},
  author={Wang, Minjie and Yu, Lingfan and Zheng, Da and Gan, Quan and Gai, Yu and Ye, Zihao and Li, Mufei and Zhou, Jinjing and Huang, Qi and Ma, Chao and others},
  year={2019}
}
```

import dgl
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from model import *
from sensors2graph import *
import torch.nn as nn
import argparse
import scipy.sparse as sp
import os

import torch.multiprocessing as mp
import torch.distributed as dist

import time






def train(gpu,args):
    rank = args.nr * args.gpus + gpu
    cuda_string = 'cuda'+':'+str(gpu)
    device = torch.device(cuda_string) if torch.cuda.is_available() and not args.disablecuda else torch.device("cpu")

    if args.pems_bay:
        # graph_pkl_filename = "./data/adj_mx_bay.pkl"
        graph_pkl_filename = "/root/workspace/hdd/data_dirs/gnnmark/stgcn/adj_mx_bay.pkl"
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        sp_mx = sp.coo_matrix(adj_mx)
    else:
        # sensor_filename = "./data/graph_sensor_ids.txt"
        sensor_filename = "/root/workspace/hdd/data_dirs/gnnmark/stgcn/graph_sensor_ids.txt"
        with open(sensor_filename) as f:
            sensor_ids = f.read().strip().split(',')
        # distance_filename = "./data/distances_la_2012.csv"
        distance_filename = "/root/workspace/hdd/data_dirs/gnnmark/stgcn/distances_la_2012.csv"
        distance_df = pd.read_csv(distance_filename, dtype={'from': 'str', 'to': 'str'})
        adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
        sp_mx = sp.coo_matrix(adj_mx)


    G = dgl.from_scipy(sp_mx)

    df = pd.read_hdf(args.tsfilepath)
    num_samples, num_nodes = df.shape

    tsdata = df.to_numpy()
    n_his = args.window

    save_path = args.savemodelpath
    n_pred = args.pred_len
    n_route = num_nodes
    blocks = args.channels

    drop_prob = 0
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    W = adj_mx
    len_val = round(num_samples * 0.1)
    len_train = round(num_samples * 0.7)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    x_train, y_train = data_transform(train, n_his, n_pred, device)
    x_val, y_val = data_transform(val, n_his, n_pred, device)
    x_test, y_test = data_transform(test, n_his, n_pred, device)

    if args.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)


    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False,sampler=train_sampler)


    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size)

    loss = nn.MSELoss()

    G = G.to(device)
    model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, args.control_str).to(device)
 
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    min_val_loss = np.inf

    for epoch in range(1, epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
         

        scheduler.step()
        
        if args.world_size > 1:
          torch.distributed.barrier()

        if rank == 0: 
          val_loss = evaluate_model(model, loss, val_iter,device)
          if val_loss < min_val_loss:
            min_val_loss = val_loss

          MAE, MAPE, RMSE = evaluate_metric(model, test_iter, scaler)
          
   
    if args.world_size > 1: 
      torch.distributed.barrier()

       
def main():
    parser = argparse.ArgumentParser(description='STGCN_WAVE')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--disablecuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for training and validation (default: 50)')
    parser.add_argument('--epochs', type=int, default=50, help='epochs for training  (default: 50)')
    parser.add_argument('--num_layers', type=int, default=9, help='number of layers')
    parser.add_argument('--window', type=int, default=144, help='window length')
    parser.add_argument('--tsfilepath', type=str, default='./data/metr-la.h5', help='ts file path')
    parser.add_argument('--savemodelpath', type=str, default='stgcnwavemodel.pt', help='save model path')
    parser.add_argument('--pred_len', type=int, default=5, help='how many steps away we want to predict')
    parser.add_argument('--control_str', type=str, default='TNTSTNTST', help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')
    parser.add_argument('--channels', type=int, nargs='+', default=[1, 16, 32, 64, 32, 128], help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of nodes (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--pems_bay', default=False, type=bool, help='Train on PEMS Bay dataset')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '129.10.52.124'
    os.environ['MASTER_PORT'] = '8888'

    if args.gpus > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0,args)    
    

if __name__ == '__main__':
 main()

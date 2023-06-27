import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from k_gnn import DataLoader, GraphConv, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin




import os

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

class MyFilter(object):
    def __call__(self, data):
        return not (data.num_nodes == 7 and data.num_edges == 12) and \
            data.num_nodes < 450


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                '1-2-3-PROTEINS')
dataset = TUDataset(
    path, name='PROTEINS',
    pre_transform=T.Compose([TwoMalkin(),
                             ConnectedThreeMalkin()]), pre_filter=MyFilter())

perm = torch.randperm(len(dataset), dtype=torch.long)
dataset = dataset[perm]

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = dataset.data.iso_type_2.max().item() + 1
dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2,
                                    num_classes=num_i_2).to(torch.float)

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = F.one_hot(dataset.data.iso_type_3,
                                    num_classes=num_i_3).to(torch.float)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64 + num_i_2, 64)
        self.conv5 = GraphConv(64, 64)
        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = data.x
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(gpu,args):
    acc = []
    rank = args.nr * args.gpus + gpu

    if args.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    cuda_string = 'cuda'+':'+str(gpu)
    device = torch.device(cuda_string) if torch.cuda.is_available()  else torch.device("cpu")
    model = Net().to(device)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    trained = False

    for i in range(10):
        if args.world_size > 1:
            model.module.reset_parameters()
        else:
            model.reset_parameters()

            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

        test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        n = len(dataset) // 10
        test_mask[i * n:(i + 1) * n] = 1
        test_dataset = dataset[test_mask]
        train_dataset = dataset[1 - test_mask]

        n = len(train_dataset) // 10
        val_mask = torch.zeros(len(train_dataset), dtype=torch.uint8)

        val_mask[i * n:(i + 1) * n] = 1
        val_dataset = train_dataset[val_mask]
        train_dataset = train_dataset[1 - val_mask]

        val_loader = DataLoader(val_dataset, batch_size=args.batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch)


        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)



        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False,sampler= train_sampler)

        


        print('---------------- Split {} ----------------'.format(i))

        best_val_loss, test_acc = 100, 0

        for epoch in range(1, args.epochs):
            lr = scheduler.optimizer.param_groups[0]['lr']
            model.train()
            loss_all = 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = F.nll_loss(model(data), data.y)
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                optimizer.step()

            train_loss = loss_all / len(train_loader.dataset)
            
            if args.world_size > 1:
              torch.distributed.barrier()

            

            if rank == 0:
                model.eval()
                loss_all = 0
                for data in val_loader:
                    data = data.to(device)
                    loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()


                val_loss = loss_all / len(val_loader.dataset)
                scheduler.step(val_loss)

                if best_val_loss >= val_loss:
                    best_val_loss = val_loss


                model.eval()
                correct = 0

                for data in test_loader:
                    data = data.to(device)
                    pred = model(data).max(1)[1]
                    correct += pred.eq(data.y).sum().item()


                test_acc = correct / len(test_loader.dataset)    
                acc.append(test_acc)

                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
              'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                  epoch, lr, train_loss, val_loss, test_acc))

               
    if args.world_size > 1:
              torch.distributed.barrier()        
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', default=False)
    parser.add_argument('--batch',type=int, default=32)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of nodes (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs',type=int, default=100)
    
    args = parser.parse_args()

    
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = ''
    os.environ['MASTER_PORT'] = ''

    if args.gpus > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0,args) 


if __name__ == '__main__':
  main()

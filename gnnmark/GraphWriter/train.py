import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from graphwriter import *
from utils import *
from opts import *
import os
import sys


import torch.multiprocessing as mp
import torch.distributed as dist

import time



def train_one_epoch(model, dataloader, optimizer, args, epoch):
    model.train()
    tloss = 0.
    tcnt = 0.
    st_time = time.time()
    with tqdm(dataloader, desc='Train Ep '+str(epoch), mininterval=60) as tq:
        for batch in tq:
            pred = model(batch)
            nll_loss = F.nll_loss(pred.view(-1, pred.shape[-1]), batch['tgt_text'].view(-1), ignore_index=0)
            loss = nll_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss = loss.item()
            if loss!=loss:
                raise ValueError('NaN appear')
            tloss += loss * len(batch['tgt_text'])
            tcnt += len(batch['tgt_text'])
            tq.set_postfix({'loss': tloss/tcnt}, refresh=False)
    print('Train Ep ', str(epoch), 'AVG Loss ', tloss/tcnt, 'Steps ', tcnt, 'Time ', time.time()-st_time, 'GPU', torch.cuda.max_memory_cached()/1024.0/1024.0/1024.0)
    torch.save(model, args.save_model+str(epoch%100))
         
         

val_loss = 2**31
def eval_it(model, dataloader, args, epoch):
    global val_loss
    model.eval()
    tloss = 0.
    tcnt = 0.
    st_time = time.time()
    with tqdm(dataloader, desc='Eval Ep '+str(epoch), mininterval=60) as tq:
        for batch in tq:
            with torch.no_grad():
                pred = model(batch)
                nll_loss = F.nll_loss(pred.view(-1, pred.shape[-1]), batch['tgt_text'].view(-1), ignore_index=0)
            loss = nll_loss
            loss = loss.item()
            tloss += loss * len(batch['tgt_text'])
            tcnt += len(batch['tgt_text'])
            tq.set_postfix({'loss': tloss/tcnt}, refresh=False)
    print('Eval Ep ', str(epoch), 'AVG Loss ', tloss/tcnt, 'Steps ', tcnt, 'Time ', time.time()-st_time)
    if tloss/tcnt < val_loss:
        print('Saving best model ', 'Ep ', epoch, ' loss ', tloss/tcnt)
        torch.save(model, args.save_model+'best')
        val_loss = tloss/tcnt



def train(gpu,args):
    rank = args.nr * args.gpus + gpu
    cuda_string = 'cuda'+':'+str(gpu)
    device = torch.device(cuda_string if torch.cuda.is_available() else 'cpu')

    if args.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    train_dataset, valid_dataset, test_dataset = get_datasets(args.fnames, device=device, save=args.save_dataset)
    args = vocab_config(args, train_dataset.ent_vocab, train_dataset.rel_vocab, train_dataset.text_vocab, train_dataset.ent_text_vocab, train_dataset.title_vocab)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler = BucketSampler(train_dataset, batch_size=args.batch_size),collate_fn=train_dataset.batch_fn)

    if args.world_size > 1:
      custom_sampler = BucketSampler(train_dataset, batch_size=args.batch_size)
      ddp_sampler =  DistributedSamplerWrapper(custom_sampler,args.gpus,rank)
      train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler = ddp_sampler,collate_fn=train_dataset.batch_fn)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, \
                    shuffle=False, collate_fn=train_dataset.batch_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, \
                    shuffle=False, collate_fn=train_dataset.batch_fn)

    model = GraphWriter(args,device)
    model.to(device)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

  
    
    for epoch in range(args.epoch):
      train_one_epoch(model, train_dataloader, optimizer, args, epoch)
      if args.world_size > 1:
        torch.distributed.barrier()
      if rank == 0:      
        eval_it(model, valid_dataloader, args, epoch)
    
    if args.world_size > 1:
      torch.distributed.barrier()         

def main():
    parser = argparse.ArgumentParser(description='Graph Writer in DGL')
    parser.add_argument('--nhid', default=500, type=int, help='hidden size')
    parser.add_argument('--nhead', default=4, type=int, help='number of heads')
    parser.add_argument('--head_dim', default=125, type=int, help='head dim')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--prop', default=6, type=int, help='number of layers of gnn')
    parser.add_argument('--title', action='store_true', help='use title input')
    parser.add_argument('--test', action='store_true', help='inference mode')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--beam_size', default=4, type=int, help='beam size, 1 for greedy')
    parser.add_argument('--epoch', default=200, type=int, help='training epoch')
    parser.add_argument('--beam_max_len', default=200, type=int, help='max length of the generated text')
    parser.add_argument('--enc_lstm_layers', default=2, type=int, help='number of layers of lstm')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    # args.add_argument('--lr_decay', default=1e-8, type=float, help='')
    parser.add_argument('--clip', default=1, type=float, help='gradient clip')
    parser.add_argument('--emb_drop', default=0.0, type=float, help='embedding dropout')
    parser.add_argument('--attn_drop', default=0.1, type=float, help='attention dropout')
    parser.add_argument('--drop', default=0.1, type=float, help='dropout')
    parser.add_argument('--lp', default=1.0, type=float, help='length penalty')
    parser.add_argument('--graph_enc', default='gtrans', type=str,
                      help='gnn mode, we only support the graph transformer now')
    parser.add_argument('--train_file', default='data/unprocessed.train.json', type=str, help='training file')
    parser.add_argument('--valid_file', default='data/unprocessed.val.json', type=str, help='validation file')
    parser.add_argument('--test_file', default='data/unprocessed.test.json', type=str, help='test file')
    parser.add_argument('--save_dataset', default='data.pickle', type=str, help='save path of dataset')
    parser.add_argument('--save_model', default='saved_model.pt', type=str, help='save path of model')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of nodes (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args.dec_ninp = args.nhid * 3 if args.title else args.nhid * 2
    args.fnames = [args.train_file, args.valid_file, args.test_file]

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = ''
    os.environ['MASTER_PORT'] = ''

    if args.world_size > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0,args)    


if __name__ == '__main__':
    main() 

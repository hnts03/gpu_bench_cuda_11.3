import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from ckpt_util import save_ckpt
import logging
import time
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import time


def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

def train_one_epoch(model, device, loader, optimizer, task_type):
    loss_list = []
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
    return statistics.mean(loss_list)


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    return evaluator.eval(input_dict)


def train(gpu,args,dataset):

    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    rank = args.nr * args.gpus + gpu
    

    if args.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    

    cuda_string = 'cuda'+':'+str(gpu)

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size,
                                   args.feature)


    device = torch.device(cuda_string) if torch.cuda.is_available()  else torch.device("cpu")

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)  
 
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset[split_idx["train"]],
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, sampler = train_sampler)
   

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DeeperGCN(args).to(device)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')

        loss_list = []
        model.train()

        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                pred = model(batch)
                is_labeled = batch.y == batch.y
                if "classification" in dataset.task_type:
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
                    loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

        epoch_loss = statistics.mean(loss_list)

        if args.world_size > 1:
          torch.distributed.barrier()

        if rank == 0 :
            train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
            valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
            test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]
            
            if args.world_size > 1:
                model.module.print_params(epoch=epoch)
            else:
                model.print_params(epoch=epoch)    

            if train_result > results['highest_train']:       
                results['highest_train'] = train_result

            if valid_result > results['highest_valid']:
                results['highest_valid'] = valid_result
                results['final_train'] = train_result
                results['final_test'] = test_result

                save_ckpt(model, optimizer,
                      round(epoch_loss, 4), epoch,
                      args.model_save_path,
                      sub_dir, name_post='valid_best')
                logging.info("%s" % results)
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))

    if args.world_size > 1:
        torch.distributed.barrier()



def main():
    args = ArgsInit().save_exp()
    procs = []

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size,
                                   args.feature)
    dataset = PygGraphPropPredDataset(name=args.dataset)
    args.num_tasks = dataset.num_tasks
    logging.info('%s' % args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    if args.world_size > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,dataset),join=True)
    elif args.world_size == 1:
        train(0,args,dataset)



if __name__ == "__main__":
    main()

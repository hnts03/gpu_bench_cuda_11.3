import torch
import numpy as np
import pandas as pd
import pickle



def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1]
    l = len(data)
    num = l-n_his-n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])
    
    cnt = 0
    for i in range(l-n_his-n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)



def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


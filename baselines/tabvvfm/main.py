import os
from time import time
from tqdm import tqdm
from typing import *
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import delu, json
import argparse

# from tqdm import tqdm
# from copy import deepcopy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from tabsyn.baselines.tabvfm.flow_matching import *
from .flow_matching import *
from .networks import *
from .preprocess import *
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler

def main(args):

    delu.random.seed(args.seed)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    # config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.train = True
    # raw_config = src.load_config(config_path)

    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    data_train = pd.read_csv(f'{real_data_path}/train.csv')

    discrete_columns = [data_train.columns[i] for i in info['cat_col_idx']]
    numerical_columns = [data_train.columns[i] for i in info['num_col_idx']]
    if info['task_type'] == 'binclass': discrete_columns += data_train.columns[info['target_col_idx']].tolist()
    else: numerical_columns += data_train.columns[info['target_col_idx']].tolist()
    
    # quantile = StandardScaler()
    quantile = QuantileTransformer()
    x_cont = quantile.fit_transform(data_train[numerical_columns])

    encoder = OneHotEncoder()
    x_cat_dummy = encoder.fit_transform(data_train[discrete_columns]).toarray()
    K = [len(data_train[k].unique()) for k in discrete_columns]

    d_cont = x_cont.shape[1] if x_cont is not None else 0 
    d_cat = x_cat_dummy.shape[1]
    d_total = d_cont + d_cat

    n_epochs = 10000
    batch_size = 4096
    # batch_size = 8192
    print(f'uses batch size {batch_size}')

    # n_epochs = 10
    
    dataset = torch.cat([torch.from_numpy(x_cont), torch.from_numpy(x_cat_dummy)], dim=1).float() if d_cont > 0 else torch.from_numpy(x_cat_dummy).float()
    dataset = dataset.to(device)
    dataset = TensorDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle = True)

    '''
        the details are in
        https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb
        or in the flow matching notebook in drive
    '''

    model = TabFlowMatching(d_cont, K)
    net = Net(d_total, 512).to(device) ## already according to tabsyn paper
    match args.cond_vel:
        case 'ot': net_t = OT_t().to(device)
        case 'vp': net_t = VPDiffusion_t().to(device)
        case 've': net_t = VEDiffusion_t().to(device)
        case 'logit': net_t = LogitNormal_t().to(device)
        case 'cos': net_t = Cosine_t().to(device)
        case _: raise Exception(f'Unknown conditional velocity formula: {args.cond_vel}, should be between "ot", "vp" and "ve"')
    
    # 
    v_t = CondVF(net, net_t, d_cont = d_cont, cat_list = K)
    losses = []
    per_batch_losses = []
    # configure optimizer
    optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=False)
    
    # in diffusion models, the optimizer is usually AdamW with weight decay
    # self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)

    print(''.center(30,'#'))
    print('Training starts'.center(30,'-'))

    best_loss = float('inf')
    patience = 0
    for epoch in range(n_epochs): # tqdm(range(n_epochs))
        batch_loss = 0.0
        len_input = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_1 = batch[0].to(device)
            # compute loss
            loss = model.loss(v_t, x_1)
            batch_loss += loss.item() * x_1.size(0)
            len_input += x_1.size(0)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(v_t.parameters(), 1.0)
            optimizer.step()
            losses += [[epoch, loss.detach().cpu().numpy()]]
        
        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(v_t.net.state_dict(), f'{model_save_path}/vf_{args.cond_vel}.pt')
        else:
            patience += 1
            # if patience == 500:
            #     print('Early stopping')
            #     break

        if (epoch+1) % args.every == 0:
            if not os.path.exists(model_save_path+'/per_epoch'):
                os.makedirs(model_save_path+'/per_epoch')
            torch.save(v_t.net.state_dict(), f'{model_save_path}/per_epoch/vf_{args.cond_vel}_{epoch+1}.pt')
        
    # torch.save(v_t.net.state_dict(), f'{model_save_path}/vf.pt')
    with open(f"{model_save_path}/vf_loss.pkl", "wb") as file:
        pickle.dump(losses, file)

    print('Training finished'.center(30,'-'))
    print(''.center(30,'#'))

    loss1 = pd.DataFrame(losses)
    loss1 = loss1.groupby(0).mean()

    plt.plot(loss1.index.values, loss1[1].values)
    plt.yscale("log")
    plt.savefig(f"{model_save_path}/loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0, help='GPU device number.')

    args = parser.parse_args()


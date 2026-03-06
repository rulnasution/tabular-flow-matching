import os
from time import time
from tqdm import tqdm
from typing import *
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import delu, json, argparse, time
import torchsde

# from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
from .flow_matching import *
from .networks import *
from .preprocess import *
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler

def main(args):

    delu.random.seed(args.seed)
    def sampling_data_ode(n_samples, t = 1.0, batch_size = 10000):
        print(f'ODE integration until t={t} using {args.steps} steps using {args.int_method}')
        # Sampling
        v_t.eval()
        n_times = (n_samples // batch_size) + 1
        # epoch_iterator = tqdm(range(n_times))
        x_1_hat = []
        for epoch in range(n_times):
            # with torch.no_grad():
            x_0 = torch.randn(batch_size, d_total, device=device)
            # x_1_hat.append(v_t.decode_t0_t1(x_0, 0.0, t).detach())
            x_1_hat.append(v_t.decode_t0_t1(x_0, 0.0, t, method = args.int_method).detach())
            # x_1_hat.append(v_t.decode(x_0).detach())
        v_t.train()

        x_1_hat = torch.cat(x_1_hat, dim = 0)
        x_1_hat = x_1_hat[:n_samples,:]
        x_1_cont = quantile.inverse_transform(x_1_hat[:,:d_cont].cpu().numpy())
        x_1_oh = torch.tensor([], device=x_1_hat.device)
        a = d_cont + 0
        for i in K:
            b = a+i
            x_1_oh = torch.cat([x_1_oh, F.one_hot(torch.argmax(x_1_hat[:,a:b],dim=1), num_classes=i)], dim = 1)
            a += i
        x_1_cont = pd.DataFrame(x_1_cont, columns=numerical_columns)
        x_1_cat = pd.DataFrame(encoder.inverse_transform(x_1_oh.cpu().numpy()), columns=discrete_columns)
        x_1 = pd.concat([x_1_cont, x_1_cat], axis=1)
        return x_1[data_train.columns]
    
        # return prepare_tddpm_to_ctgan_prepr(torch.tanh(x_1_hat[:n_samples,:d_cont]).cpu().numpy() if d_cont != 0 else None,
        #                                     x_1_hat[:n_samples,d_cont:].cpu().numpy(),
        #                                     tran_prep, tran_info,prep_list_num, prep_list_cat)
    
    def sampling_data_sde(n_samples, t = 1.0, batch_size = 10000):
        print(f'SDE integration until t={t} using {args.steps} steps using {args.int_method}')
        # Sampling
        v_t.eval()
        n_times = (n_samples // batch_size) + 1
        ts = torch.linspace(0, t, 2).to('cuda:0')
        # epoch_iterator = tqdm(range(n_times))
        x_1_hat = []
        if args.int_method in ['heun', 'midpoint']: v_t.sde_type = "stratonovich"
        for epoch in range(n_times):
            with torch.no_grad():
                x_0 = torch.randn(batch_size, d_total, device=device)
                res1 = torchsde.sdeint(v_t, x_0, ts, method=args.int_method, dt = 1/args.steps)
                x_1_hat.append(res1[-1].detach())
            # x_1_hat.append(v_t.decode(x_0).detach())
        v_t.train()

        x_1_hat = torch.cat(x_1_hat, dim = 0)
        x_1_hat = x_1_hat[:n_samples,:]
        x_1_cont = quantile.inverse_transform(x_1_hat[:,:d_cont].cpu().numpy())
        x_1_oh = torch.tensor([], device=x_1_hat.device)
        a = d_cont + 0
        for i in K:
            b = a+i
            x_1_oh = torch.cat([x_1_oh, F.one_hot(torch.argmax(x_1_hat[:,a:b],dim=1), num_classes=i)], dim = 1)
            a += i
        x_1_cont = pd.DataFrame(x_1_cont, columns=numerical_columns)
        x_1_cat = pd.DataFrame(encoder.inverse_transform(x_1_oh.cpu().numpy()), columns=discrete_columns)
        x_1 = pd.concat([x_1_cont, x_1_cat], axis=1)
        return x_1[data_train.columns]
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'
    save_path = args.save_path

    config_path = f'{curr_dir}/configs/{dataname}.toml'
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

    # tran_prep, tran_info = prepare_data_trans_ctgan(data_train, discrete_columns)
    # x_cont, x_cat, x_cat_dummy, prep_list_num, prep_list_cat = prepare_ctgan_prepr_to_tddpm(data_train,tran_prep, tran_info)
    # K = [k.dim for k in prep_list_cat]

    # quantile = StandardScaler()
    quantile = QuantileTransformer()
    quantile.fit(data_train[numerical_columns])

    encoder = OneHotEncoder()
    encoder.fit(data_train[discrete_columns])
    K = [len(data_train[k].unique()) for k in discrete_columns]

    d_cont = len(numerical_columns) 
    d_cat = sum(K)
    d_total = d_cont + d_cat


    print('Start sampling...')
    start_time = time.time()
    
    net = Net(d_total, 512).to(device) ## already according to tabsyn paper
    if args.saved_epoch == 0: ## don't forget to add _{args.cond_vel}
        print(f'tabvfm uses {args.cond_vel} trajectory and best model')
        net.load_state_dict(torch.load(f'{model_save_path}/vf_{args.cond_vel}.pt'))
    else:
        print(f'tabvfm uses {args.cond_vel} trajectory and {args.saved_epoch} epoch model')
        net.load_state_dict(torch.load(f'{model_save_path}/per_epoch/vf_{args.cond_vel}_{args.saved_epoch}.pt'))
    
    match args.cond_vel:
        case 'ot': net_t = OT_t().to(device)
        case 'vp': net_t = VPDiffusion_t().to(device)
        case 've': net_t = VEDiffusion_t().to(device)
        case 'logit': net_t = LogitNormal_t().to(device)
        case 'cos': net_t = Cosine_t().to(device)
        case _: raise Exception(f'Unknown conditional velocity formula: {args.cond_vel}, should be between "ot", "vp" and "ve", "logit", and "cos"')

    if not args.sde:
        print('Sampling using ODE')
        v_t = CondVF(net, net_t, d_cont = d_cont, cat_list = K, n_steps = args.steps).to(device)
        syn_df = sampling_data_ode(len(data_train), t = args.t_ode, batch_size= len(data_train) if args.batch_size == 0 else args.batch_size)
    else:
        match args.cond_vel_sigma:
            case 'ot': net_t_sigma = OT_t().to(device)
            case 'vp': net_t_sigma = VPDiffusion_t().to(device)
            case 've': net_t_sigma = VEDiffusion_t().to(device)
            case 'logit': net_t_sigma = LogitNormal_t().to(device)
            case 'cos': net_t_sigma = Cosine_t().to(device)
            case _: raise Exception(f'Unknown sigma formula: {args.cond_vel}, should be between "ot", "vp" and "ve", "logit", and "cos"')

        print('Sampling using SDE')
        v_t = StochasticCondVF(net, net_t, net_t_sigma, sigma_max = args.sigma_max, d_cont = d_cont, cat_list = K)
        
        '''since torchsde cannot overcome the Inf problem when we divide by beta, 
           we need to clip the t manually when t -> 1'''
        if args.cond_vel in ['vp', 'cos'] and args.t_ode == 1.:
            t_target = args.t_ode - 1e-5
        else:
            t_target = args.t_ode
        syn_df = sampling_data_sde(len(data_train), t = t_target, batch_size= len(data_train) if args.batch_size == 0 else args.batch_size)
    
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print(f'Sampling time = {end_time - start_time}')
    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabVFM')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')

    args = parser.parse_args()
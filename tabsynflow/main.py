import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
# from tabsynflow.model import MLPDiffusion, Model
from .latent_utils import get_input_train
from .fm_utils import *

import delu

warnings.filterwarnings('ignore')


def main(args): 
    delu.random.seed(args.seed) ## default 42
    device = args.device

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = 10000 + 1 # 10000 + 1
    
    # num_epochs = 10

    match args.cond_vel:
        case 'ot': model = OTFlowMatching()
        case 'vp': model = VPDiffusionFlowMatching()
        case 've': model = VEDiffusionFlowMatching()
        case 'logit': model = LogitFlowMatching()
        case 'cos': model = CosineFlowMatching()
        case _: raise Exception(f'Unknown conditional velocity formula: {args.cond_vel}, should be between "ot", "vp" and "ve", "logit", and "cos" ')

    print(f'training using {args.cond_vel} conditional velocity formula')
    # net = Net(in_dim, in_dim, [1024, 2048, 2048, 1024], 10).to(device)
    net = Net(in_dim, 512).to(device)
    v_t = CondVF(net)
    num_params = sum(p.numel() for p in v_t.net.parameters())
    print("the number of parameters", num_params)

    losses = []
    # per_batch_losses = []
    # configure optimizer
    # optimizer = torch.optim.AdamW(v_t.parameters(), lr=1e-3)

    optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=False)

    v_t.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            # loss = model(inputs)
            
            # loss = loss.mean()
            loss = model.loss(v_t, inputs)
            
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(v_t.net.state_dict(), f'{ckpt_path}/model_{args.cond_vel}.pt')
        else:
            patience += 1
            # if patience == 500:
            #     print('Early stopping')
            #     break

        if epoch % 1000 == 0:
            torch.save(v_t.net.state_dict(), f'{ckpt_path}/model_{args.cond_vel}_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSynFlow')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
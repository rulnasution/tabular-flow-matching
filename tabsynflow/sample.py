import torch

import argparse
import warnings
import time

# from tabsyn.model import MLPDiffusion, Model
# from tabsyn.diffusion_utils import sample
from .latent_utils import get_input_generate, recover_data, split_num_cat_target
from .fm_utils import Net, CondVF 
import delu
from tqdm import tqdm

warnings.filterwarnings('ignore')


def main(args):
    delu.random.seed(args.seed) ## default 42
    dataname = args.dataname
    device = args.device
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]
    print(train_z.shape)

    mean = train_z.mean(0)

    def sampling_data(n_samples, t = 1.0, batch_size = 10000):
        # Sampling
        print(f'ODE integration until t={t} using {args.int_method}')
        n_times = (n_samples // batch_size) + 1
        epoch_iterator = tqdm(range(n_times))
        x_1_hat = []
        for epoch in range(n_times):
            with torch.no_grad():
                # x_0 = torch.cat([torch.randn(n_samples, d_cont, device=device),
                #                 sample_cat(n_samples, K)], dim=1)
                x_0 = torch.randn(batch_size, in_dim, device=device)
                # x_1_hat.append(v_t.decode(x_0))
                # x_1_hat.append(v_t.decode_t0_t1(x_0, 0.0, t))
                x_1_hat.append(v_t.decode_t0_t1(x_0, 0.0, t, method = args.int_method))

        x_1_hat = torch.cat(x_1_hat, dim = 0)
        return x_1_hat[:n_samples,:]

    # denoise_fn = MLPDiffusion(in_dim, 1024).to(device)    
    # model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    # model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    # net = Net(in_dim, in_dim, [1024], 10).to(device)
    net = Net(in_dim, 512).to(device)
    if args.saved_epoch == 0: ## don't forget to add _{args.cond_vel}
        print(f'Tabsynflow uses {args.cond_vel} trajectory and best model')
        net.load_state_dict(torch.load(f'{ckpt_path}/model_{args.cond_vel}.pt'))
    else:
        print(f'Tabsynflow uses {args.cond_vel} trajectory and {args.saved_epoch} epoch model')
        net.load_state_dict(torch.load(f'{ckpt_path}/model_{args.cond_vel}_{args.saved_epoch}.pt'))
    
    v_t = CondVF(net, n_steps=args.steps)
    # v_t.net.load_state_dict(torch.load(f'{ckpt_path}/model_{args.cond_vel}.pt'))

    
    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = sampling_data(num_samples, args.t_ode, 20000)
    # x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
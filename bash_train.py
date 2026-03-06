import subprocess
import json
import argparse
import pandas as pd
import os

def run_command(command, env_name=None, output=False):
    """ Runs a shell command in a specified Conda environment. """
    if env_name:
        command = f'conda run -n {env_name} {command}'
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True, capture_output=output,text=True)
    if output:
        if result.returncode == 0:
            # Get the last non-empty line of stdout
            last_line = result.stdout.strip().split("\n")[-1]
            return last_line
        else:
            print(f"Error: {result.stderr}")
            return None
    else:
        return result

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TabVFM')
    # parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--dataname', type=int, default=0, help='Name of dataset.')
    parser.add_argument('--method', type=str, default='tabvfm', help='Algorithm used.')
    parser.add_argument('--n_eval', type=int, default=1, help='number of evaluations.')

    args = parser.parse_args()
    
    datanames = ['adult','beijing','default','magic','news','shoppers', 
                 'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
    idx = args.dataname-1
    dataname = datanames[idx]
    # print("\n Activating first environment: tabsyn\n")
    
    ## running the model
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    if args.method=='tabvfm':
        run_command(f"python main.py --dataname {dataname} --method {args.method} --mode train --every 1000 --size 0 --width 0 --batch_size 4096 --seed 42", env_name="tabsyn")
    elif 'tabsyn' in args.method:
        if not os.path.exists(f'{curr_dir}/tabsyn/vae/ckpt/{dataname}/train_z.npy'):
            run_command(f"python main.py --dataname {dataname} --method vae --mode train", env_name="tabsyn")
        else:
            print('vae has been trained already')
        
        # if not os.path.exists(f'{curr_dir}/{args.method}/ckpt/{dataname}/model.pt'):
        run_command(f"python main.py --dataname {dataname} --method {args.method} --mode train", env_name="tabsyn")
        # else:
        #     print(f'{args.method} has been trained already')
    else:
        run_command(f"python main.py --dataname {dataname} --method {args.method} --mode train", env_name="tabsyn")

        # if not os.path.exists(f'{curr_dir}/baselines/{args.method}/ckpt/{dataname}/model.pt'):
        #     run_command(f"python main.py --dataname {dataname} --method {args.method} --mode train", env_name="tabsyn")
        # else:
        #     print(f'{args.method} has been trained already')

    if args.method == 'tabvfm':
        for s_epoch in range(0, 10000, 1000): # range(0, 10000, 1000): # range(0, 4000, 1000)  
            for s in range(args.n_eval):
                run_command(f'python main.py --dataname {dataname} --method {args.method} --mode sample --size 0 --width 0 --batch_size 20000 --seed {s_epoch+s} --steps 100 --t_ode 1 --saved_epoch {s_epoch} --save_path "synthetic/{dataname}/{args.method}_{s_epoch}_{s}.csv"', env_name="tabsyn")
    else:
        for s in range(args.n_eval):
            run_command(f'python main.py --dataname {dataname} --method {args.method} --mode sample --batch_size 20000 --seed {s} --save_path "synthetic/{dataname}/{args.method}_{s}.csv"', env_name="tabsyn")
            # print("\n Switching to second environment: synthcity\n")
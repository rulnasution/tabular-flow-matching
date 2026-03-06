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
    
    final_res = []
    range_epoch = [0] if args.method != 'tabvfm' else [i for i in range(0, 1000, 1000)]
    for s_epoch in range_epoch: # range(0, 1000, 100) 
        for s in range(args.n_eval):
            # print("\n Switching to second environment: synthcity\n")
            syn_file = f'{args.method}_{s}' if args.method != 'tabvfm' else f'{args.method}_{s_epoch}_{s}'
            
            run_command(f'python eval/eval_urisk.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            with open(f'eval/urisk/{dataname}/{args.method}.json', "r") as file:
                res_urisk = json.load(file)
            
            keys = list(res_urisk.keys())
            extracted_values = list(res_urisk.values())
            
            # print(keys, extracted_values)

            final_res.append(extracted_values)
            
            # print("\n✅ Script finished!")

        final_results = pd.DataFrame(final_res, columns=keys)
        save_dir = f'eval/combine/{dataname}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_results.to_csv(f"{save_dir}/{args.method}_urisk.csv", index=False)
    print(f"Final Results saved to final_results_{dataname}_{args.method}_urisk.csv")
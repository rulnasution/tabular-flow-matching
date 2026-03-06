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
    
    datanames = ['adult','beijing','default','magic','news','shoppers']
    idx = args.dataname-1
    dataname = datanames[idx]
    # print("\n Activating first environment: tabsyn\n")
    
    final_res = []
    range_epoch = [0] if args.method != 'tabvfm' else [i for i in range(0, 10000, 1000)]
    for s_epoch in range_epoch: # range(0, 1000, 100) 
        for s in range(args.n_eval):
            # print("\n Switching to second environment: synthcity\n")
            syn_file = f'{args.method}_{s}' if args.method != 'tabvfm' else f'{args.method}_{s_epoch}_{s}'
            
            run_command(f'python eval/eval_density.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            with open(f"eval/density/{dataname}/{args.method}/quality.txt", "r") as file:
                shape_val, trend_val = map(float, file.read().splitlines())
            print(f"Extracted Values: Column Shapes: {shape_val}, Column Pair Trends: {trend_val}")    

            
            run_command(f'python eval/eval_mle.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            with open(f'eval/mle/{dataname}/{args.method}.json', "r") as file:
                res_mle = json.load(file)
            # print(res_mle)
            if idx in [0,2,3,5]: best_mle_data = res_mle.get("best_auroc_scores", {})
            else: best_mle_data = res_mle.get('best_rmse_scores', {}) 
            # Convert the values into a single list
            
            keys = []
            extracted_values = []
            for classifier, metrics in best_mle_data.items():
                keys.extend(metrics.keys())
                extracted_values.extend(metrics.values())  #
            
            print(keys, extracted_values)

            detection_val = run_command(f'python eval/eval_detection.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
            print(f"Detection Value: {detection_val}")

            # dcr_val = run_command(f"python eval/eval_dcr.py --dataname {args.dataname} --model {args.method}", env_name="tabsyn", output=True)
            dcr_val = run_command(f'python eval/eval_dcr.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
            print(f"DCR Value: {dcr_val}")
            
            # dcr_val = 0.0

            final_keys = ['epoch','sample_seed',"shape", "trend", "detection", "dcr"] + keys
            final_values = [s_epoch, s, shape_val, trend_val, detection_val, dcr_val] + extracted_values
            final_res.append(final_values)
            
            # print("\n✅ Script finished!")

        final_results = pd.DataFrame(final_res, columns=final_keys)
        save_dir = f'eval/combine/{dataname}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_results.to_csv(f"{save_dir}/{args.method}.csv", index=False)
    print(f"Final Results saved to final_results_{dataname}_{args.method}.csv")
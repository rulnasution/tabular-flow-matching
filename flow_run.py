import subprocess
import json
import argparse
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument('--cond_vel', type=str, default='ot', help='conditional velocity')
    parser.add_argument('--train', action='store_true', default=False, help='training or not.')
    parser.add_argument('--sample', action='store_true', default=False, help='sample or not.')
    parser.add_argument('--saved_epoch', type=int, default=0, help="using mid-epoch, set 0 if you want to use latest model")
    parser.add_argument('--eval_other_dir', type=str, default='', help="directory eval (from tabsyn paper), '' means not evaluated")
    parser.add_argument('--eval_quality_dir', type=str, default='', help='directory eval of quality (from tabsyn paper), '' means not evaluated')
    parser.add_argument('--eval_urisk_dir', type=str, default='', help='directory eval of utility and risk, '' means not evaluated')
    parser.add_argument('--t_ode', type=float, default=1., help='Stop the ODE at time t during synthesis')
    parser.add_argument('--steps', type=int, default=100, help='NFEs. for VFM is the number of discrete steps for ODE integration (Actually NFEs as well)')
    parser.add_argument('--sde', action = 'store_true', default=False, help='Whether using SDE or ODE. Dont add this argument if you want ODE')
    parser.add_argument('--cond_vel_sigma', type=str, default='ot', help='Sigma term for SDE. Between ot, vp, ve, logit, and cos')
    parser.add_argument('--sigma_max', type=float, default=1., help='Maximum of the sigma value allowed, usually multiplication with the sigma term')
    
    args = parser.parse_args()
    
    datanames = ['adult','beijing','default','magic','news','shoppers', 
                 'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn','tcga','diabetes','tcgaa']
    idx = args.dataname-1
    dataname = datanames[idx]
    # print("\n Activating first environment: tabsyn\n")
    
    ## running the model
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    if args.train:
        if 'tabsyn' in args.method:
            if not os.path.exists(f'{curr_dir}/tabsyn/vae/ckpt/{dataname}/train_z.npy'):
                run_command(f"python main.py --dataname {dataname} --method vae --mode train", env_name="tabsyn")
            else:
                print('vae has been trained already')
            
        run_command(f"python main.py --dataname {dataname} --method {args.method} --mode train --cond_vel {args.cond_vel}", env_name="tabsyn")

    if args.sample:
        sde_note = '--sde' if args.sde else ''
        sde_sigma = f'_{args.cond_vel_sigma}' if args.sde else ''

        for s in range(args.n_eval):
            run_command(f'python main.py --dataname {dataname} --method {args.method} --mode sample --batch_size 20000 --seed {s} --saved_epoch {args.saved_epoch} --t_ode {args.t_ode} --cond_vel {args.cond_vel} --steps {args.steps} {sde_note} --cond_vel_sigma {args.cond_vel_sigma} --sigma_max {args.sigma_max} --save_path "synthetic/{dataname}/{args.method}_{s}_{args.cond_vel}{sde_sigma}.csv"', env_name="tabsyn")
    
    if args.eval_other_dir != '':   
        final_res = []
        for s in range(args.n_eval):
            sde_sigma = f'_{args.cond_vel_sigma}' if args.sde else ''

            syn_file = f'{args.method}_{s}_{args.cond_vel}{sde_sigma}'
            
            run_command(f'python eval/eval_density.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            with open(f"eval/density/{dataname}/{args.method}/quality.txt", "r") as file:
                shape_val, trend_val = map(float, file.read().splitlines())
            print(f"Extracted Values: Column Shapes: {shape_val}, Column Pair Trends: {trend_val}")    

            final_keys = ['sample_seed',"shape", "trend"]
            final_values = [s, shape_val, trend_val] 
            final_res.append(final_values)
            
            # detection_val = run_command(f'python eval/eval_detection.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
            # print(f"Detection Value: {detection_val}")
                
            # final_keys = ['sample_seed',"shape", "trend", "detection"]
            # final_values = [s, shape_val, trend_val, detection_val] 
            # final_res.append(final_values)
            
            # run_command(f'python eval/eval_mle.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            # with open(f'eval/mle/{dataname}/{args.method}.json', "r") as file:
            #     res_mle = json.load(file)
            # # print(res_mle)
            # if idx in [0,2,3,5]: best_mle_data = res_mle.get("best_auroc_scores", {})
            # else: best_mle_data = res_mle.get('best_rmse_scores', {}) 
            # # Convert the values into a single list
            
            # keys = []
            # extracted_values = []
            # for classifier, metrics in best_mle_data.items():
            #     keys.extend(metrics.keys())
            #     extracted_values.extend(metrics.values())
            
            # print(keys, extracted_values)

            # detection_val = run_command(f'python eval/eval_detection.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
            # print(f"Detection Value: {detection_val}")

            # # dcr_val = run_command(f"python eval/eval_dcr.py --dataname {args.dataname} --model {args.method}", env_name="tabsyn", output=True)
            # dcr_val = run_command(f'python eval/eval_dcr.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
            # print(f"DCR Value: {dcr_val}")
            
            # dcr_val = 0.0

            # final_keys = ['sample_seed',"shape", "trend", "detection", "dcr"] + keys
            # final_values = [s, shape_val, trend_val, detection_val, dcr_val] + extracted_values
            # final_res.append(final_values)
            
            # print("\n✅ Script finished!")
            # os.remove(f"synthetic/{dataname}/{syn_file}.csv")

        final_results = pd.DataFrame(final_res, columns=final_keys)
        save_dir = f'eval/combine/{dataname}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_results.to_csv(f"{save_dir}/{args.eval_other_dir}_other.csv", index=False)
        print(f"Final Results saved to {save_dir}/{args.eval_other_dir}_other.csv")

    if args.eval_quality_dir != '':
        final_res = []
        for s in range(args.n_eval):

            sde_sigma = f'_{args.cond_vel_sigma}' if args.sde else ''
            syn_file = f'{args.method}_{s}_{args.cond_vel}{sde_sigma}'
            
            if dataname == 'diabetes':
                run_command(f'python eval/eval_quality_diabetes.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="synthcity")    
                with open(f"eval/quality/{dataname}/{args.method}.txt", "r") as file:
                    alpha_pr, beta_re, wd, mmd = map(float, file.read().splitlines())
                print(f"Extracted Values: alpha precision: {alpha_pr}, beta recall: {beta_re}")
                # alpha_pr, beta_re = 0, 0
                
                final_keys = ['sample_seed', 'alpha_pr', 'beta_re', 'wd', 'mmd']
                final_values = [s, alpha_pr, beta_re, wd, mmd]
                final_res.append(final_values)
            elif dataname in ['canada', 'fiji', 'uk', 'rwanda', 'indonesia', 'adulta','churn', 'tcga']:
                run_command(f'python eval/eval_quality_distance.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="synthcity")    
                with open(f"eval/quality/{dataname}/{args.method}.txt", "r") as file:
                    alpha_pr, beta_re, wd, mmd = map(float, file.read().splitlines())
                print(f"Extracted Values: alpha precision: {alpha_pr}, beta recall: {beta_re}, WD: {wd}, MMD: {mmd}")
                # alpha_pr, beta_re = 0, 0
                
                final_keys = ['sample_seed', 'alpha_pr', 'beta_re', 'wd', 'mmd']
                final_values = [s, alpha_pr, beta_re, wd, mmd]
                final_res.append(final_values)    
            else:
                run_command(f'python eval/eval_quality.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="synthcity")    
                with open(f"eval/quality/{dataname}/{args.method}.txt", "r") as file:
                    alpha_pr, beta_re = map(float, file.read().splitlines())
                print(f"Extracted Values: alpha precision: {alpha_pr}, beta recall: {beta_re}")
                # alpha_pr, beta_re = 0, 0
                
                final_keys = ['sample_seed', 'alpha_pr', 'beta_re']
                final_values = [s, alpha_pr, beta_re]
                final_res.append(final_values)    

            
            # print("\n✅ Script finished!")
            if args.dataname not in ['tcga','diabetes']:
                os.remove(f"synthetic/{dataname}/{syn_file}.csv")

        final_results = pd.DataFrame(final_res, columns=final_keys)
        save_dir = f'eval/combine/{dataname}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_results.to_csv(f"{save_dir}/{args.eval_quality_dir}_quality.csv", index=False)
        print(f"Quality Results saved to {save_dir}/{args.eval_quality_dir}_quality.csv")

    if args.eval_urisk_dir != '':
        final_res = []
        for s in range(args.n_eval):
            # print("\n Switching to second environment: synthcity\n")
            sde_sigma = f'_{args.cond_vel_sigma}' if args.sde else ''
            syn_file = f'{args.method}_{s}_{args.cond_vel}{sde_sigma}'
            
            run_command(f'python eval/eval_urisk.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            with open(f'eval/urisk/{dataname}/{args.method}.json', "r") as file:
                res_urisk = json.load(file)
            
            if dataname != 'tcga':
                run_command(f'python eval/eval_density.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")
            else:
                run_command(f'python eval/eval_density_tcga.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn")

            with open(f"eval/density/{dataname}/{args.method}/quality.txt", "r") as file:
                shape_val, trend_val = map(float, file.read().splitlines())
            print(f"Extracted Values: Column Shapes: {shape_val}, Column Pair Trends: {trend_val}")    

            try:
                detection_val = run_command(f'python eval/eval_detection.py --dataname {dataname} --model {args.method} --path "synthetic/{dataname}/{syn_file}.csv"', env_name="tabsyn", output=True)
                print(f"Detection Value: {detection_val}")
            except Exception as e:
                print(f'error in detection run {e}')
                detection_val = 0
            keys = list(res_urisk.keys()) + ["shape", "trend", "detection"]
            extracted_values = list(res_urisk.values()) + [shape_val, trend_val, detection_val]
            
            
            ### Use this for only utility
            
            # keys = list(res_urisk.keys())
            # extracted_values = list(res_urisk.values())
            final_res.append(extracted_values)
            
            os.remove(f"synthetic/{dataname}/{syn_file}.csv")

        final_results = pd.DataFrame(final_res, columns=keys)
        save_dir = f'eval/combine/{dataname}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_results.to_csv(f"{save_dir}/{args.eval_urisk_dir}_urisk.csv", index=False)
        print(f"Final Results saved to {save_dir}/{args.eval_urisk_dir}_urisk.csv")
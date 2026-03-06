import numpy as np
import pandas as pd
import os 
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_train import preprocess, TabularDataset
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

pd.options.mode.chained_assignment = None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')


args = parser.parse_args()


if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path
    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 

    print(syn_path)
    

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    if dataname in ['canada', 'fiji', 'uk', 'rwanda', 'indonesia', 'adulta','churn','tcga','tcgaa','diabetes']:

        syn_data = pd.read_csv(syn_path, dtype=str)
        real_data = pd.read_csv(real_path, dtype=str)

        discrete_columns = [real_data.columns[i] for i in info['cat_col_idx']]
        numerical_columns = [real_data.columns[i] for i in info['num_col_idx']]
        if info['task_type'] == 'binclass': discrete_columns += real_data.columns[info['target_col_idx']].tolist()
        else: numerical_columns += real_data.columns[info['target_col_idx']].tolist()

        real_data[numerical_columns] = real_data[numerical_columns].astype(float)
        if dataname not in ['adulta', 'churn','tcga','tcgaa','diabetes']:
            real_data['AGE'] = real_data['AGE'].round(0).astype(int)
        
        # real_data[discrete_columns] = real_data[discrete_columns].replace('nan', np.nan, inplace=True)
        real_data[discrete_columns] = real_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                        value=r'\1', regex=True)
        
        syn_data[numerical_columns] = syn_data[numerical_columns].astype(float)
        if dataname not in ['adulta', 'churn','tcga','tcgaa','diabetes']:
            syn_data['AGE'] = syn_data['AGE'].round(0).astype(int)
        # syn_data[discrete_columns] = syn_data[discrete_columns].replace('nan', np.nan, inplace=True)
        syn_data[discrete_columns] = syn_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                        value=r'\1', regex=True)
        syn_data.fillna('nan')

    else:
        syn_data = pd.read_csv(syn_path)
        real_data = pd.read_csv(real_path)

    ''' Special treatment for default dataset and CoDi model '''

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx
        
    num_real_data = real_data[num_col_idx]
    # cat_real_data2_np = real_data[cat_col_idx].to_numpy().astype('str')
    cat_real_data = real_data[cat_col_idx].drop([16,17,18], axis=1)

    # for a in cat_col_idx:
    #     print(a, len(real_data[a].unique()))
    # raise Exception('babi')

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
    
    num_syn_data = syn_data[num_col_idx]
    # cat_syn_data2_np = syn_data[cat_col_idx].to_numpy().astype('str')
    cat_syn_data = syn_data[cat_col_idx].drop([16,17,18], axis=1)

    num_syn_data_np = num_syn_data.to_numpy()

    # cat_syn_data_np = np.array
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
    if (dataname == 'default' or dataname == 'news') and model[:4] == 'codi':
        cat_syn_data_np = cat_syn_data.astype('int').to_numpy().astype('str')

    elif model[:5] == 'great':
        if dataname == 'shoppers':
            cat_syn_data_np[:, 1] = cat_syn_data[11].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 2] = cat_syn_data[12].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 3] = cat_syn_data[13].astype('int').to_numpy().astype('str')
            
            max_data = cat_real_data[14].max()
        
            cat_syn_data.loc[cat_syn_data[14] > max_data, 14] = max_data
            # cat_syn_data[14] = cat_syn_data[14].apply(lambda x: threshold if x > max_data else x)
            
            cat_syn_data_np[:, 4] = cat_syn_data[14].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 4] = cat_syn_data[14].astype('int').to_numpy().astype('str')
        
        elif dataname in ['default', 'faults', 'beijing']:

            columns = cat_real_data.columns
            for i, col in enumerate(columns):
                if (cat_real_data[col].dtype == 'int'):

                    max_data = cat_real_data[col].max()
                    min_data = cat_real_data[col].min()

                    cat_syn_data.loc[cat_syn_data[col] > max_data, col] = max_data
                    cat_syn_data.loc[cat_syn_data[col] < min_data, col] = min_data

                    cat_syn_data_np[:, i] = cat_syn_data[col].astype('int').to_numpy().astype('str')
                    
        else:
            cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    else:
        cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)


    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    chunk_size = len(real_data)  # adjust based on available memory

    le_real_data_chunks = [pd.DataFrame(np.concatenate((num_real_data_np[i:i+chunk_size], cat_real_data_oh[i:i+chunk_size]), axis=1)).astype(float) 
                           for i in range(0, len(num_real_data_np), chunk_size)]
    le_real_num_chunks = [pd.DataFrame(num_real_data_np[i:i+chunk_size]).astype(float) 
                          for i in range(0, len(num_real_data_np), chunk_size)]
    le_real_cat_chunks = [pd.DataFrame(cat_real_data_oh[i:i+chunk_size]).astype(float) 
                          for i in range(0, len(cat_real_data_oh), chunk_size)]

    le_syn_data_chunks = [pd.DataFrame(np.concatenate((num_syn_data_np[i:i+chunk_size], cat_syn_data_oh[i:i+chunk_size]), axis=1)).astype(float) 
                          for i in range(0, len(num_syn_data_np), chunk_size)]
    le_syn_num_chunks = [pd.DataFrame(num_syn_data_np[i:i+chunk_size]).astype(float) 
                         for i in range(0, len(num_syn_data_np), chunk_size)]
    le_syn_cat_chunks = [pd.DataFrame(cat_syn_data_oh[i:i+chunk_size]).astype(float) 
                         for i in range(0, len(cat_syn_data_oh), chunk_size)]

    np.set_printoptions(precision=4)
    result = []

    print('=========== All Features ===========')
    print('Data shape: ', (len(num_real_data_np), le_real_data_chunks[0].shape[1]) if le_real_data_chunks else 'empty')

    qual_scores = []
    for idx in range(len(le_real_data_chunks)):
        print(f'Processing chunk {idx+1}/{len(le_real_data_chunks)}')
        X_syn_loader = GenericDataLoader(le_syn_data_chunks[idx])
        X_real_loader = GenericDataLoader(le_real_data_chunks[idx])

        quality_evaluator = eval_statistical.AlphaPrecision()
        qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
        qual_res = {
            k: v for (k, v) in qual_res.items() if "naive" in k
        }
        qual_scores.append(qual_res)

    # average across chunks
    avg_alpha_precision = np.mean([res['delta_precision_alpha_naive'] for res in qual_scores])
    avg_beta_recall = np.mean([res['delta_coverage_beta_naive'] for res in qual_scores])

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(avg_alpha_precision, avg_beta_recall))

    Alpha_Precision_all = avg_alpha_precision
    Beta_Recall_all = avg_beta_recall

    save_dir = f'eval/quality/{dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # with open(f'{save_dir}/{model}.txt', 'w') as f:
    #     f.write(f'{Alpha_Precision_all}\n')
    #     f.write(f'{Beta_Recall_all}\n')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)
    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis = 1)).astype(float)
    le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)

    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis = 1)).astype(float)
    le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

    print('=========== All Features ===========')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    wd = 0.0
    mmd = 0.0

    wd_eval = eval_statistical.WassersteinDistance()
    mmd_eval = eval_statistical.MaximumMeanDiscrepancy(kernel="rbf")

    batch_size = 5000
    seed = 0

    # 1) samakan panjang, lalu shuffle SEKALI, lalu batch berurutan
    n = min(len(le_real_data), len(le_syn_data))
    real_shuf = le_real_data.iloc[:n].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    syn_shuf  = le_syn_data.iloc[:n].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_batches = n // batch_size  # full batches only

    wd_scores, mmd_scores = [], []

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size

        real_batch = real_shuf.iloc[start:end]
        syn_batch  = syn_shuf.iloc[start:end]

        X_real_loader = GenericDataLoader(real_batch)
        X_syn_loader  = GenericDataLoader(syn_batch)

        wd_out  = wd_eval.evaluate_default(X_real_loader, X_syn_loader)
        mmd_out = mmd_eval.evaluate_default(X_real_loader, X_syn_loader)

        # output bisa float atau dict tergantung versi
        wd_val  = float(wd_out["score"]) if isinstance(wd_out, dict) and "score" in wd_out else float(wd_out)
        mmd_val = float(mmd_out["score"]) if isinstance(mmd_out, dict) and "score" in mmd_out else float(mmd_out)

        wd_scores.append(wd_val)
        mmd_scores.append(mmd_val)

        print(f"Batch {b+1}/{n_batches} | WD={wd_val:.6f} | MMD={mmd_val:.6f}")

    print("WD mean/std:", float(np.mean(wd_scores)), float(np.std(wd_scores)))
    print("MMD mean/std:", float(np.mean(mmd_scores)), float(np.std(mmd_scores)))
    wd = float(np.mean(wd_scores))
    mmd = float(np.mean(mmd_scores))

    with open(f'{save_dir}/{model}.txt', 'w') as f:
        f.write(f'{Alpha_Precision_all}\n')
        f.write(f'{Beta_Recall_all}\n')
        f.write(f'{wd}\n')
        f.write(f'{mmd}\n')
    # if os.path.isfile(f'{save_dir}/{model}.txt'):
    #     with open(f'{save_dir}/{model}.txt', 'a') as f:
    #         f.write(f'{Alpha_Precision_all}, {Beta_Recall_all}\n')
    # else:
    #     with open(f'{save_dir}/{model}.txt', 'w') as f:
    #         f.write(f'{Alpha_Precision_all}, {Beta_Recall_all}\n')

import numpy as np
import pandas as pd
import os 

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from itertools import combinations
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity

from sdmetrics.errors import ConstantInputError
import numpy as np


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata

def _is_constant(s):
    # works for pandas Series
    x = s.to_numpy()
    x = x[~np.isnan(x)] if np.issubdtype(x.dtype, np.floating) else x
    return x.size == 0 or np.all(x == x[0])

def _const_score(real_s, syn_s):
    real_const = _is_constant(real_s)
    syn_const  = _is_constant(syn_s)
    if real_const and syn_const:
        # both constant: perfect only if same constant value (ignoring NaNs)
        rv = real_s.dropna().iloc[0] if real_s.dropna().shape[0] else np.nan
        sv = syn_s.dropna().iloc[0] if syn_s.dropna().shape[0] else np.nan
        return 1.0 if rv == sv else 0.0
    # one constant, the other not: mismatch in variability
    return 0.0


def compute_shape_trend_blockwise(new_real_data, new_syn_data, n_num, block_size=100):
    """
    Returns: Shape (float), Trend (float)
    - Shape: avg KSComplement (numerical) + TVComplement (categorical/boolean)
    - Trend: avg pair scores, computed block-wise:
        * cat-cat once (ContingencySimilarity)
        * num-cat for each block (ContingencySimilarity with discretization)
        * num-num within block only (CorrelationSimilarity)
    """
    n_total = new_real_data.shape[1]
    cont_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_total))

    # ---------- Shape ----------
    shape_scores = []
    for i in range(n_total):
        if i < n_num:
            s = KSComplement.compute(new_real_data[i], new_syn_data[i])
        else:
            s = TVComplement.compute(new_real_data[i], new_syn_data[i])
        shape_scores.append(float(s))
    Shape = float(np.mean(shape_scores)) if shape_scores else float("nan")

    # ---------- Trend (block-wise) ----------
    trend_scores = []

    # cat-cat once
    for c1, c2 in combinations(cat_idx, 2):
        s = ContingencySimilarity.compute(
            real_data=new_real_data[[c1, c2]],
            synthetic_data=new_syn_data[[c1, c2]],
        )
        trend_scores.append(float(s))

    # blocks over numerical cols, each block + all categorical
    for start in range(0, len(cont_idx), block_size):
        block = cont_idx[start:start + block_size]

        # num-cat (for this block)
        for ncol in block:
            for ccol in cat_idx:
                s = ContingencySimilarity.compute(
                    real_data=new_real_data[[ncol, ccol]],
                    synthetic_data=new_syn_data[[ncol, ccol]],
                    # continuous_column_names=[ncol],  # discretize numerical then contingency
                )
                trend_scores.append(float(s))

        # num-num within block only
        if len(block) >= 2:
            for c1, c2 in combinations(block, 2):
                r1, r2 = new_real_data[c1], new_real_data[c2]
                s1, s2 = new_syn_data[c1], new_syn_data[c2]

                if _is_constant(s1) or _is_constant(s2) or _is_constant(r1) or _is_constant(r2):
                    score = _const_score(r1, s1) * _const_score(r2, s2)  # conservative
                    trend_scores.append(float(score))
                    continue

                try:
                    score = CorrelationSimilarity.compute(
                        real_data=new_real_data[[c1, c2]],
                        synthetic_data=new_syn_data[[c1, c2]],
                        coefficient="Pearson",
                    )
                    trend_scores.append(float(score))
                except ConstantInputError:
                    trend_scores.append(0.0)


    Trend = float(np.mean(trend_scores)) if trend_scores else float(0)
    return Shape, Trend


if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 
    # print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    if dataname in ['canada', 'fiji', 'uk', 'rwanda', 'indonesia', 'adulta','churn','tcga','diabetes']:

        syn_data = pd.read_csv(syn_path, dtype=str)
        real_data = pd.read_csv(real_path, dtype=str)

        discrete_columns = [real_data.columns[i] for i in info['cat_col_idx']]
        numerical_columns = [real_data.columns[i] for i in info['num_col_idx']]
        if info['task_type'] == 'binclass': discrete_columns += real_data.columns[info['target_col_idx']].tolist()
        else: numerical_columns += real_data.columns[info['target_col_idx']].tolist()

        real_data[numerical_columns] = real_data[numerical_columns].astype(float)
        if dataname not in ['adulta', 'churn','tcga','diabetes']:
            real_data['AGE'] = real_data['AGE'].round(0).astype(int)
        
        # real_data[discrete_columns] = real_data[discrete_columns].replace('nan', np.nan, inplace=True)
        real_data[discrete_columns] = real_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                        value=r'\1', regex=True)
        
        syn_data[numerical_columns] = syn_data[numerical_columns].astype(float)
        if dataname not in ['adulta', 'churn','tcga','diabetes']:
            syn_data['AGE'] = syn_data['AGE'].round(0).astype(int)
        # syn_data[discrete_columns] = syn_data[discrete_columns].replace('nan', np.nan, inplace=True)
        syn_data[discrete_columns] = syn_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                        value=r'\1', regex=True)
        syn_data.fillna('nan')

    else:
        syn_data = pd.read_csv(syn_path)
        real_data = pd.read_csv(real_path)

    save_dir = f'eval/density/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    # n_num = number of numerical columns after your preprocessing
    # you already built numerical_columns earlier, including target if regression
    n_num = len(numerical_columns)

    # compute Shape/Trend fast (block-wise trend)
    Shape, Trend = compute_shape_trend_blockwise(
        new_real_data, new_syn_data, n_num=n_num, block_size=5
    )

    with open(f'{save_dir}/quality.txt', 'w') as f:
        f.write(f'{Shape}\n')
        f.write(f'{Trend}\n')

    Quality = (Shape + Trend) / 2

    # DiagnosticReport (keep as before)
    diag_report = DiagnosticReport()
    diag_report.generate(new_real_data, new_syn_data, metadata)
    coverages = diag_report.get_details('Coverage')
    coverages.to_csv(f'{save_dir}/coverage.csv', index=False)


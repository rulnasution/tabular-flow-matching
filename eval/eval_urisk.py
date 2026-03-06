import numpy as np
import pandas as pd
import os 

import json

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm
from sklearn.preprocessing import KBinsDiscretizer

import ot  # POT library
import numpy as np


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def eval_syn_data(name, orig, syn):
    country_name = name.upper() if name=='uk' else name.capitalize()
    country_name = 'Adult' if name == 'adulta' else country_name
    roc_val = cal_mean_roc(country_name,orig,syn)
    cio_val = cal_mean_cio(country_name,orig,syn)
    if country_name not in ['Adult', 'Churn', 'Diabetes', 'Tcga']:
        orig['AGE'] = orig['AGE'].astype(str)
        syn['AGE'] = syn['AGE'].astype(str)
    tcap_val = cal_mean_tcap(country_name,orig,syn)
    evaluation = {'ROC_uni': roc_val[0], 'ROC_biv': roc_val[1], 'CIO': cio_val, 'TCAP': tcap_val, 
    'Utility':(roc_val[0]+roc_val[1]+cio_val)/3, 'Risk': max(0, tcap_val)}
    # evaluation = [roc_val[0], roc_val[1], cio_val, tcap_val, (roc_val[0]+roc_val[1]+cio_val)/3, max(0, tcap_val)]
    return evaluation

def cal_mean_cio(name,orig,syn):
    if name == 'UK':
        target_cols = ['TENURE','MSTATUS']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
        orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
        syn['MSTATUS'] = (syn['MSTATUS'] == 'Married' ) | (syn['MSTATUS'] == 'Remarried' )
        syn['TENURE'] = (syn['TENURE'] == 'Own occ-buying' ) | (syn['TENURE'] == 'Own occ-outright' )
    elif name=='Canada':
        target_cols = ['TENURE','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['ABIDENT','CLASSWK','DEGREE','EMPSTAT','SEX','URBAN','TENURE','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE', 'HRSWK', 'INCTOT', 'WKSWORK']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a4' )| (orig['MARST'] == '2' ) | (orig['MARST'] == '4' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' ) | (orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a4' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '4' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Fiji':
        target_cols = ['TENURE','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['CLASSWKR','ETHNIC','RELIGION','EDATTAIN','SEX','PROV','TENURE','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' )|(orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' )|(orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Rwanda':
        target_cols = ['OWNERSH','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['DISAB1','EDCERT','CLASSWK','LIT','RELIG','SEX','OWNERSH','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['OWNERSH'] =((orig['OWNERSH'] == 'a1' ) | (orig['OWNERSH'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['OWNERSH'] = ((syn['OWNERSH'] == 'a1' ) | (syn['OWNERSH'] == '1' )).astype('int')
    elif name=='Indonesia':
        target_cols = ['OWNERSHIP','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['LANDOWN', 'RELATE', 'SEX', 'HOMEFEM', 'HOMEMALE', 'RELIGION', 
                    'LIT', 'SCHOOL', 'EDATTAIND', 'DISABLED','OWNERSHIP','MARST']
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        
        orig['MARST'] = ((orig['MARST'] == '3' ) | (orig['MARST'] == '4' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == 'a4' )).astype('int')
        orig['OWNERSHIP'] =((orig['OWNERSHIP'] == 'a1' ) | (orig['OWNERSHIP'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == '3' ) | (syn['MARST'] == '4' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == 'a4' )).astype('int')
        syn['OWNERSHIP'] = ((syn['OWNERSHIP'] == 'a1' ) | (syn['OWNERSHIP'] == '1' )).astype('int')
    elif name=='Adult':
        target_cols = ['income','marital-status']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['workclass', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
        cont_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['income'] = ((orig['income'] == '>50K' ) | (orig['income'] == '>50K.' )).astype('int')
        orig['marital-status'] =((orig['marital-status'] == 'Married-civ-spouse' ) | (orig['marital-status'] == 'Married-spouse-absent' ) | (orig['marital-status'] == 'Married-AF-spouse')).astype('int')
        syn['income'] = ((syn['income'] == '>50K' ) | (syn['income'] == '>50K.' )).astype('int')
        syn['marital-status'] =((syn['marital-status'] == 'Married-civ-spouse' ) | (syn['marital-status'] == 'Married-spouse-absent' ) | (syn['marital-status'] == 'Married-AF-spouse')).astype('int')
    elif name=='Churn':
        target_cols = ['Exited','CreditScore','EstimatedSalary']
        families = [sm.families.Binomial(),sm.families.Gaussian(),sm.families.Gaussian()]
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
        cont_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['Exited'] = ((orig['Exited'] == '1' )).astype('int')
        syn['Exited'] = ((syn['Exited'] == '1' )).astype('int')
    elif name=='Insurance':
        target_cols = ['charges']
        families = [sm.families.Gaussian()]
        
        key_cols = ['sex', 'children', 'smoker', 'region']
        cont_cols = ['charges','age','bmi']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]

        orig = orig[key_cols]
        syn = syn[key_cols]
        # median_cutoff = orig['charges'].median()
        # orig['charges'] = ((orig['charges'] > median_cutoff )).astype('int')
        # syn['charges'] = ((syn['charges'] > median_cutoff )).astype('int')
    elif name=='Credit':
        target_cols = ['checking_balance', 'default']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
        cont_cols = ['months_loan_duration', 'amount', 'age']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]

        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['checking_balance'] = ((orig['checking_balance'] == '1 - 200 DM') | (orig['checking_balance'] == '> 200 DM')).astype('int')
        orig['credit_history'] =((orig['credit_history'] == 'repaid') | (orig['credit_history'] == 'fully repaid') | (orig['credit_history'] == 'fully repaid this bank')).astype('int')
        orig['savings_balance'] = ((orig['savings_balance'] == '501 - 1000 DM' ) | (orig['savings_balance'] == '> 1000 DM')).astype('int')
        syn['checking_balance'] = ((syn['checking_balance'] == '1 - 200 DM') | (syn['checking_balance'] == '> 200 DM')).astype('int')
        syn['credit_history'] =((syn['credit_history'] == 'repaid' ) | (syn['credit_history'] == 'fully repaid') | (syn['credit_history'] == 'fully repaid this bank')).astype('int')
        syn['savings_balance'] =((syn['savings_balance'] == '1 - 200 DM') | (syn['savings_balance'] == '> 1000 DM') ).astype('int')
    
    elif name=='Diabetes':
        target_cols = ['change','diabetesMed','readmitted']
        families = [sm.families.Binomial(),sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['race', 'gender', 'age', 'weight', 'diag_1',
                    'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
                    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                    'tolazamide', 'examide', 'citoglipton', 'insulin',
                    'glyburide-metformin', 'glipizide-metformin',
                    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                    'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']
        cont_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures','num_medications',
                     'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['change'] = ((orig['change'] == 'Ch' )).astype('int')
        syn['change'] = ((syn['change'] == 'Ch' )).astype('int')

        orig['diabetesMed'] = ((orig['diabetesMed'] == 'Yes' )).astype('int')
        syn['diabetesMed'] = ((syn['diabetesMed'] == 'Yes' )).astype('int')

        orig['readmitted'] = ((orig['readmitted'] != 'NO' )).astype('int')
        syn['readmitted'] = ((syn['readmitted'] != 'NO' )).astype('int')
    elif name=='Tcga':
        target_cols = ['Class']
        key_cols = ['Class']
        families = [sm.families.Binomial()]
        cont_cols = orig.drop('Class',axis=1).columns.tolist()

        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]

        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['Class'] = ((orig['Class'] == 'KIRC' )).astype('int')
        syn['Class'] = ((syn['Class'] == 'KIRC' )).astype('int')


    orig.fillna('a0',inplace=True)
    syn.fillna('a0',inplace=True)
    encoder = OrdinalEncoder()
    # encoder.fit(pd.concat([orig.astype(str),syn.astype(str)],axis=0))
    encoder.fit(pd.concat([orig.astype(str)],axis=0))
    orig = pd.DataFrame(encoder.transform(orig.astype(str)),columns=key_cols)
    syn = pd.DataFrame(encoder.transform(syn.astype(str)),columns=key_cols)
    if len(cont_cols) > 0:
        orig = pd.concat([orig,orig_cont],axis=1)
        syn = pd.concat([syn,syn_cont],axis=1)
        key_cols += cont_cols

    scores = []
    for target, family in zip(target_cols, families):
        orig_glm = sm.GLM(orig[target].astype(float),orig.drop(columns=target).astype(float),family=family)
        syn_glm = sm.GLM(syn[target].astype(float),syn.drop(columns=target).astype(float),family=family)
        results = CIO_function(orig_glm, syn_glm)
        scores.append(results['mean_ci_overlap_noNeg'])
    return np.mean(scores)

# def cal_mean_cio(name,orig,syn):
#     if name == 'UK':
#         target_cols = ['TENURE','MSTATUS']
#         key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
#         orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
#         syn['MSTATUS'] = (syn['MSTATUS'] == 'Married' ) | (syn['MSTATUS'] == 'Remarried' )
#         syn['TENURE'] = (syn['TENURE'] == 'Own occ-buying' ) | (syn['TENURE'] == 'Own occ-outright' )
#     elif name=='Canada':
#         target_cols = ['TENURE','MARST']
#         key_cols = ['ABIDENT','AGE','CLASSWK','DEGREE','EMPSTAT','SEX','URBAN','TENURE','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a4' )| (orig['MARST'] == '2' ) | (orig['MARST'] == '4' )).astype('int')
#         orig['TENURE'] =((orig['TENURE'] == 'a1' ) | (orig['TENURE'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a4' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '4' )).astype('int')
#         syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
#     elif name=='Fiji':
#         target_cols = ['TENURE','MARST']
#         key_cols = ['AGE','CLASSWKR','ETHNIC','RELIGION','EDATTAIN','SEX','PROV','TENURE','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' )|(orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
#         orig['TENURE'] =((orig['TENURE'] == 'a1' )|(orig['TENURE'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
#         syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
#     elif name=='Rwanda':
#         target_cols = ['OWNERSH','MARST']
#         key_cols = ['AGE','DISAB1','EDCERT','CLASSWK','LIT','RELIG','SEX','OWNERSH','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
#         orig['OWNERSH'] =((orig['OWNERSH'] == 'a1' ) | (orig['OWNERSH'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
#         syn['OWNERSH'] = ((syn['OWNERSH'] == 'a1' ) | (syn['OWNERSH'] == '1' )).astype('int')
#     elif name=='Indonesia':
#         target_cols = ['OWNERSHIP','MARST']
#         key_cols = ['LANDOWN', 'AGE', 'RELATE', 'SEX', 'HOMEFEM', 'HOMEMALE', 'RELIGION', 
#                     'LIT', 'SCHOOL', 'EDATTAIND', 'DISABLED','OWNERSHIP','MARST']
#         orig = orig[key_cols]
#         syn = syn[key_cols]
        
#         orig['MARST'] = ((orig['MARST'] == '3' ) | (orig['MARST'] == '4' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == 'a4' )).astype('int')
#         orig['OWNERSHIP'] =((orig['OWNERSHIP'] == 'a1' ) | (orig['OWNERSHIP'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == '3' ) | (syn['MARST'] == '4' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == 'a4' )).astype('int')
#         syn['OWNERSHIP'] = ((syn['OWNERSHIP'] == 'a1' ) | (syn['OWNERSHIP'] == '1' )).astype('int')
#     elif name=='Adult':
#         target_cols = ['income','marital-status']
#         key_cols = ['workclass', 'education-num',
#                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                     'native-country','income']
#         ## 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['income'] = ((orig['income'] == '>50K' ) | (orig['income'] == '>50K.' )).astype('int')
#         orig['marital-status'] =((orig['marital-status'] == 'Married-civ-spouse' ) | (orig['marital-status'] == 'Married-spouse-absent' ) | (orig['marital-status'] == 'Married-AF-spouse')).astype('int')
#         syn['income'] = ((syn['income'] == '>50K' ) | (syn['income'] == '>50K.' )).astype('int')
#         syn['marital-status'] =((syn['marital-status'] == 'Married-civ-spouse' ) | (syn['marital-status'] == 'Married-spouse-absent' ) | (syn['marital-status'] == 'Married-AF-spouse')).astype('int')
#     elif name=='Churn':
#         target_cols = ['Exited']
#         key_cols = ['Geography', 'Gender', 'Tenure', 
#                     'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
#         ## 'CreditScore', 'Age', 'Balance', 'EstimatedSalary'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['Exited'] = ((orig['Exited'] == '1' )).astype('int')
#         syn['Exited'] = ((syn['Exited'] == '1' )).astype('int')
#     elif name=='Insurance':
#         target_cols = ['charges']
#         key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
#         ## 'age','bmi'
#         median_cutoff = orig['charges'].median()
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['charges'] = ((orig['charges'] > median_cutoff )).astype('int')
#         syn['charges'] = ((syn['charges'] > median_cutoff )).astype('int')
#     elif name=='Credit':
#         target_cols = ['checking_balance', 'default']
#         key_cols = ['checking_balance', 'credit_history', 'purpose',
#                     'savings_balance', 'employment_length', 'installment_rate',
#                     'personal_status', 'other_debtors', 'residence_history', 'property',
#                     'installment_plan', 'housing', 'existing_credits', 'default',
#                     'dependents', 'telephone', 'foreign_worker', 'job']
#         ## 'months_loan_duration', 'amount', 'age'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['checking_balance'] = ((orig['checking_balance'] == '1 - 200 DM') | (orig['checking_balance'] == '> 200 DM')).astype('int')
#         orig['credit_history'] =((orig['credit_history'] == 'repaid') | (orig['credit_history'] == 'fully repaid') | (orig['credit_history'] == 'fully repaid this bank')).astype('int')
#         orig['savings_balance'] = ((orig['savings_balance'] == '501 - 1000 DM' ) | (orig['savings_balance'] == '> 1000 DM')).astype('int')
#         syn['checking_balance'] = ((syn['checking_balance'] == '1 - 200 DM') | (syn['checking_balance'] == '> 200 DM')).astype('int')
#         syn['credit_history'] =((syn['credit_history'] == 'repaid' ) | (syn['credit_history'] == 'fully repaid') | (syn['credit_history'] == 'fully repaid this bank')).astype('int')
#         syn['savings_balance'] =((syn['savings_balance'] == '1 - 200 DM') | (syn['savings_balance'] == '> 1000 DM') ).astype('int')

#     orig.fillna('a0',inplace=True)
#     syn.fillna('a0',inplace=True)
#     encoder = OrdinalEncoder()
#     encoder.fit(pd.concat([orig.astype(str),syn.astype(str)],axis=0))
#     orig = pd.DataFrame(encoder.transform(orig.astype(str)),columns=key_cols)
#     syn = pd.DataFrame(encoder.transform(syn.astype(str)),columns=key_cols)
#     scores = []
#     for target in target_cols:
#         orig_glm = sm.GLM(orig[target].astype(float),orig.drop(columns=target).astype(float),family=sm.families.Binomial() )
#         syn_glm = sm.GLM(syn[target].astype(float),syn.drop(columns=target).astype(float),family=sm.families.Binomial() )
#         results = CIO_function(orig_glm, syn_glm)
#         scores.append(results['mean_ci_overlap_noNeg'])
#     return np.mean(scores)

def CIO_function(orig_glm,syn_glm):
    # # put them into a form so it is easier to extract the coefficients etc.
    try:
        syn_glm = syn_glm.fit()
        orig_glm = orig_glm.fit()
    except:
        return {'mean_std_coef_diff':0, 
                'median_std_coef_diff' : 0,
                'mean_ci_overlap':0, 
                'median_ci_overlap' : 0,
                'mean_ci_overlap_noNeg' :0, 
                'median_ci_overlap_noNeg':0}  # when there is a perfect separation in syn dataset

    syn_glm = pd.DataFrame(syn_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    orig_glm = pd.DataFrame(orig_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    syn_glm = syn_glm.iloc[:,:3] # take the first three columns
    orig_glm = orig_glm.iloc[:,:3]
    
    # join the original and synth
    combined = orig_glm.merge(syn_glm,how='left',on='names',suffixes=('_orig', '_syn'))
    for i in combined.columns[1:]:
        combined[i] = combined[i].astype('float')
    combined['std.coef_diff'] = abs(combined['Estimate_orig']-combined['Estimate_syn']) / (combined['stderr_orig'])
    combined['orig_lower'] = combined['Estimate_orig'] - 1.96 * combined['stderr_orig']
    combined['orig_upper'] = combined['Estimate_orig'] + 1.96 * combined['stderr_orig']
    combined['syn_lower'] = combined['Estimate_syn'] - 1.96 * combined['stderr_syn']
    combined['syn_upper'] = combined['Estimate_syn'] + 1.96 * combined['stderr_syn']
    combined['ci_overlap'] = 0.5 * (
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['orig_upper']-combined['orig_lower']) + 
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['syn_upper']-combined['syn_lower'])
                                    )
    for index,row in combined.iterrows():
        if row['orig_lower'] == row['orig_upper'] and row['orig_upper'] == row['syn_lower'] and row['syn_upper'] == row['syn_lower']:
            combined.loc[index,'ci_overlap'] = 1.0
    combined = combined[['names','std.coef_diff','ci_overlap']]
    
    combined.fillna(0,inplace=True) # set negative overlaps to zero
    combined['ci_overlap_noNeg'] = [0 if i<0 else i for i in combined['ci_overlap']]

    results = {'mean_std_coef_diff':combined['std.coef_diff'].mean(), 
                'median_std_coef_diff' : combined['std.coef_diff'].median(),
                'mean_ci_overlap': combined.ci_overlap.mean(), 
                'median_ci_overlap' : combined.ci_overlap.median(),
                # add in the overlaps where negatives were changed to zeros
                'mean_ci_overlap_noNeg' :combined.ci_overlap_noNeg.mean(), 
                'median_ci_overlap_noNeg':combined.ci_overlap_noNeg.median()}
    # now compute std. diff and ci overlap
    return results

# def CIO_function_cont(orig,syn):
#     # # CIO for continuous variables
#     # # make consistent form with CIO_function
#     try:
#         n = len(orig)
#         combined = pd.DataFrame({'Estimate_orig':orig.mean(),
#                         'Estimate_syn':syn.mean(),
#                         'stderr_orig':orig.std()/(n**0.5),
#                         'stderr_syn':syn.std()/(n**0.5),
#                         }).reset_index()
#         combined.rename(columns={'index': 'names'}, inplace=True)
#     except:
#         return {'mean_std_coef_diff':0, 
#                 'median_std_coef_diff' : 0,
#                 'mean_ci_overlap':0, 
#                 'median_ci_overlap' : 0,
#                 'mean_ci_overlap_noNeg' :0, 
#                 'median_ci_overlap_noNeg':0}  # when there is a perfect separation in syn dataset

#     combined['std.coef_diff'] = abs(combined['Estimate_orig']-combined['Estimate_syn']) / (combined['stderr_orig'])
#     combined['orig_lower'] = combined['Estimate_orig'] - 1.96 * combined['stderr_orig']
#     combined['orig_upper'] = combined['Estimate_orig'] + 1.96 * combined['stderr_orig']
#     combined['syn_lower'] = combined['Estimate_syn'] - 1.96 * combined['stderr_syn']
#     combined['syn_upper'] = combined['Estimate_syn'] + 1.96 * combined['stderr_syn']
#     combined['ci_overlap'] = 0.5 * (
#                                     (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
#                                     (combined['orig_upper']-combined['orig_lower']) + 
#                                     (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
#                                     (combined['syn_upper']-combined['syn_lower'])
#                                     )
#     print(combined)
#     for index,row in combined.iterrows():
#         if row['orig_lower'] == row['orig_upper'] and row['orig_upper'] == row['syn_lower'] and row['syn_upper'] == row['syn_lower']:
#             combined.loc[index,'ci_overlap'] = 1.0
#     combined = combined[['names','std.coef_diff','ci_overlap']]
    
#     combined.fillna(0,inplace=True) # set negative overlaps to zero
#     combined['ci_overlap_noNeg'] = [0 if i<0 else i for i in combined['ci_overlap']]

#     results = {'mean_std_coef_diff':combined['std.coef_diff'].mean(), 
#                 'median_std_coef_diff' : combined['std.coef_diff'].median(),
#                 'mean_ci_overlap': combined.ci_overlap.mean(), 
#                 'median_ci_overlap' : combined.ci_overlap.median(),
#                 # add in the overlaps where negatives were changed to zeros
#                 'mean_ci_overlap_noNeg' :combined.ci_overlap_noNeg.mean(), 
#                 'median_ci_overlap_noNeg':combined.ci_overlap_noNeg.median()}
#     # now compute std. diff and ci overlap
#     return results


def cal_mean_roc(name,orig,syn,bi=True):
    if name == 'UK':
        key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS',
                    'TENURE','MSTATUS']
    elif name=='Canada':
        key_cols = ['ABIDENT','SEX','TENURE','URBAN','BPLMOM','BPLPOP',
                    'CITIZEN','LANG','MARST','RELATE','MINORITY','RELIG','BPL']
    elif name=='Fiji':
        key_cols = ['PROV','TENURE','RELATE','SEX','ETHNIC','MARST',
                    'RELIGION','BPLPROV','RESPROV',
                    'RESSTAT','SCHOOL','TRAVEL']
    elif name=='Rwanda':
        key_cols = ['STATUS','SEX','URBAN','OWNERSH','DISAB2','DISAB1',
                    'RELATE','RELIG','HINS','NATION','BPL']
    elif name=='Indonesia':
        key_cols = ['OWNERSHIP', 'LANDOWN', 'RELATE', 'SEX', 'MARST', 
                    'HOMEMALE', 'RELIGION', 'SCHOOL', 'LIT', 'EDATTAIND', 'DISABLED']
    elif name=='Adult':
        key_cols = ['workclass', 'education',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
    elif name=='Churn':
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
    elif name=='Credit':
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
    elif name=='Diabetes':
        key_cols = ['race', 'gender', 'age', 'weight', 'diag_1',
                    'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
                    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                    'tolazamide', 'examide', 'citoglipton', 'insulin',
                    'glyburide-metformin', 'glipizide-metformin',
                    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                    'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']
    elif name == 'Tcga':
        key_cols = ['Class']
        bi = False
    orig = orig[key_cols]
    syn = syn[key_cols]


    # if name == 'UK':
    #     key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS',
    #                 'TENURE','MSTATUS']
    #     num_cols = ['AGE','HOURS']
    # elif name=='Canada':
    #     key_cols = ['ABIDENT','SEX','TENURE','URBAN','BPLMOM','BPLPOP',
    #                 'CITIZEN','LANG','MARST','RELATE','MINORITY','RELIG','BPL']
    #     num_cols = ['AGE','HRSWK','INCTOT','WKSWORK']
    # elif name=='Fiji':
    #     key_cols = ['PROV','TENURE','RELATE','SEX','ETHNIC','MARST',
    #                 'RELIGION','BPLPROV','RESPROV',
    #                 'RESSTAT','SCHOOL','TRAVEL']
    #     num_cols = ['AGE']
    # elif name=='Rwanda':
    #     key_cols = ['STATUS','SEX','URBAN','OWNERSH','DISAB2','DISAB1',
    #                 'RELATE','RELIG','HINS','NATION','BPL']
    #     num_cols = ['AGE']
    # elif name=='Indonesia':
    #     key_cols = ['OWNERSHIP', 'LANDOWN', 'RELATE', 'SEX', 'MARST', 
    #                 'HOMEMALE', 'RELIGION', 'SCHOOL', 'LIT', 'EDATTAIND', 'DISABLED']
    #     num_cols = ['AGE']
    # elif name=='Adult':
    #     key_cols = ['workclass', 'education',
    #                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #                 'native-country','income']
    #     num_cols = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
    # elif name=='Churn':
    #     key_cols = ['Geography', 'Gender', 'Tenure', 
    #                 'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
    #     num_cols = ['CreditScore', 'Age', 'Balance','EstimatedSalary']
    # elif name=='Insurance':
    #     key_cols = ['sex', 'children', 'smoker', 'region']
    #     num_cols = ['age','bmi','charges']
    # elif name=='Credit':
    #     key_cols = ['checking_balance', 'credit_history', 'purpose',
    #                 'savings_balance', 'employment_length', 'installment_rate',
    #                 'personal_status', 'other_debtors', 'residence_history', 'property',
    #                 'installment_plan', 'housing', 'existing_credits', 'default',
    #                 'dependents', 'telephone', 'foreign_worker', 'job']
    #     num_cols = ['months_loan_duration','amount','age']

    # kb = KBinsDiscretizer(encode='ordinal')
    # kb.fit(orig[num_cols])
    # orig[num_cols] = kb.transform(orig[num_cols]).astype(int)
    # syn[num_cols] = kb.transform(syn[num_cols]).astype(int)
    # orig = orig[key_cols+num_cols]
    # syn = syn[key_cols+num_cols]
    
    uni_scores = []
    bi_scores = []
    for i in range(len(key_cols)):
        uni_scores.append(roc_univariate(orig, syn, i) )
      
        if bi and i+1<len(key_cols):# max i == len(key_cols)-1
            for j in range(i+1,len(key_cols)):
                bi_scores.append(roc_bivariate(orig, syn, i, j))
    if bi:
        return np.mean(uni_scores),np.mean(bi_scores)
    else:
        return np.mean(uni_scores),0

def roc_univariate(original,synthetic,var_num):
    # create frequency tables for the original and synthetic data, on the variable
    orig_table = original.iloc[:,var_num].value_counts().reset_index()
    syn_table = synthetic.iloc[:,var_num].value_counts().reset_index()
    orig_table.columns = ['value','Freq']
    syn_table.columns = ['value','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variable
    combined = orig_table.merge(syn_table,on=['value'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


def roc_bivariate(original, synthetic, var1, var2):
    # create frequency tables for the original and synthetic data, on the two variable cross-tabulation
    orig_table = pd.crosstab(index=original.iloc[:,var1],columns=original.iloc[:,var2]).stack().reset_index()
    syn_table = pd.crosstab(index=synthetic.iloc[:,var1],columns=synthetic.iloc[:,var2]).stack().reset_index()
    orig_table.columns = ['Var1','Var2','Freq']
    syn_table.columns = ['Var1','Var2','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variables
    combined = orig_table.merge(syn_table,on=['Var1', 'Var2'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


'''
function:     replace_missing()   
description:  replaces missing values dependant on data type. Categorical or object NAs are replaced with 'blank', numerical NAs with -999. Can be modified as required
input:        pandas dataframe
output:       pandas dataframe with missing values replaced
'''
def replace_missing(dataset):
    # get a dictionary of the different data types
    types = dataset.dtypes.to_dict()
    # replace object or categorical NAs with 'blank', and numerical with -999
    for col_nam, typ in types.items():
        if (typ == 'O' or typ == 'c'):
            dataset[col_nam] = dataset[col_nam].fillna('blank')
        if (typ == 'float64' or typ == 'int64'):
            dataset[col_nam] = dataset[col_nam].fillna(-999)
    return(dataset)

def cal_mean_tcap(name,orig,syn):
    if name=='UK':
        target_cols = ['LTILL','FAMTYPE','TENURE']
        key_cols = ['AREAP','AGE','SEX','MSTATUS','ETHGROUP','ECONPRIM']
    elif name=='Canada':
        target_cols = ['RELIG','CITIZEN','TENURE']
        key_cols = ['AGE','SEX','MARST','MINORITY','EMPSTAT','BPL']
    elif name=='Fiji':
        target_cols = ['RELIGION','WORKTYPE','TENURE']
        key_cols = ['PROV','AGE','SEX','MARST','ETHNIC','CLASSWKR']
    elif name=='Rwanda':
        target_cols = ['RELIG','WKSECTOR','OWNERSH']
        key_cols = ['AGE','SEX','MARST','CLASSWK','URBAN','BPL']
    elif name=='Indonesia':
        target_cols = ['RELIGION','OWNERSHIP','EDATTAIND']
        key_cols = ['AGE', 'SEX', 'MARST', 'HOMEFEM','SCHOOL', 'LANDOWN']
    elif name=='Adult':
        target_cols = ['native-country', 'race', 'occupation']
        key_cols = ['education-num', 'marital-status', 'workclass', 
                    'relationship', 'sex', 'income']
    elif name=='Churn':
        target_cols = ['Geography']
        key_cols = ['Gender', 'Tenure', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        target_cols = ['children']
        key_cols = ['sex','region', 'smoker']
    elif name=='Credit':
        target_cols = ['checking_balance', 'credit_history', 'savings_balance']
        key_cols = ['purpose', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
    elif name == 'Diabetes':
        target_cols = ['race','diag_1','diag_2','diag_3']
        key_cols = ['gender', 'age', 'weight', 'max_glu_serum', 'A1Cresult',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin',
            'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone', 'change', 'diabetesMed','readmitted']
    elif name == 'Tcga':
        target_cols = ['Class']
        key_cols = orig.drop('Class',axis=1).columns.tolist()[:50]
        kb = KBinsDiscretizer(encode='ordinal')
        kb.fit(orig[key_cols])
        orig[key_cols] = kb.transform(orig[key_cols]).astype(int)
        syn[key_cols] = kb.transform(syn[key_cols]).astype(int)

    # if name=='UK':
    #     target_cols = ['LTILL','FAMTYPE','TENURE']
    #     key_cols = ['AREAP','AGE','HOURS','SEX','MSTATUS','ETHGROUP','ECONPRIM']
    #     num_cols = ['AGE','HOURS']
    # elif name=='Canada':
    #     target_cols = ['RELIG','CITIZEN','TENURE']
    #     key_cols = ['AGE','SEX','MARST','MINORITY','EMPSTAT','BPL',
    #                 'HRSWK','INCTOT','WKSWORK']
    #     num_cols = ['AGE','HRSWK','INCTOT','WKSWORK']
    # elif name=='Fiji':
    #     target_cols = ['RELIGION','WORKTYPE','TENURE']
    #     key_cols = ['PROV','AGE','SEX','MARST','ETHNIC','CLASSWKR']
    #     num_cols = ['AGE']
    # elif name=='Rwanda':
    #     target_cols = ['RELIG','WKSECTOR','OWNERSH']
    #     key_cols = ['AGE','SEX','MARST','CLASSWK','URBAN','BPL']
    #     num_cols = ['AGE']
    # elif name=='Indonesia':
    #     target_cols = ['RELIGION','OWNERSHIP','EDATTAIND']
    #     key_cols = ['AGE', 'SEX', 'MARST', 'HOMEFEM','SCHOOL', 'LANDOWN']
    #     num_cols = ['AGE']
    # elif name=='Adult':
    #     target_cols = ['native-country', 'race', 'occupation']
    #     key_cols = ['education-num', 'marital-status', 'workclass', 
    #                 'relationship', 'sex', 'income',
    #                 'age','fnlwgt','capital-gain','capital-loss','hours-per-week'
    #                 ]
    #     num_cols = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
    # elif name=='Churn':
    #     target_cols = ['Geography']
    #     key_cols = ['Gender', 'Tenure', 'NumOfProducts', 
    #                 'HasCrCard', 'IsActiveMember','Exited',
    #                 'CreditScore', 'Age', 'Balance','EstimatedSalary']
    #     num_cols = ['CreditScore', 'Age', 'Balance','EstimatedSalary']
    # elif name=='Insurance':
    #     target_cols = ['children','charges']
    #     key_cols = ['sex','region', 'smoker','age','bmi']
    #     num_cols = ['age','bmi','charges']
    # elif name=='Credit':
    #     target_cols = ['checking_balance', 'credit_history', 'savings_balance']
    #     key_cols = ['purpose', 'employment_length', 'installment_rate',
    #                 'personal_status', 'other_debtors', 'residence_history', 'property',
    #                 'installment_plan', 'housing', 'existing_credits', 'default',
    #                 'dependents', 'telephone', 'foreign_worker', 'job',
    #                 'months_loan_duration','amount','age']
    #     num_cols = ['months_loan_duration','amount','age']

    # kb = KBinsDiscretizer(n_bins=3,encode='ordinal')
    # kb.fit(orig[num_cols])
    # orig[num_cols] = kb.transform(orig[num_cols]).astype(int)
    # syn[num_cols] = kb.transform(syn[num_cols]).astype(int)
    
    scores = []
    for target in target_cols:
        for i in range(3,len(key_cols)+1):
            score,baseline = tcap(orig,syn,target,key_cols[:i],verbose=False)
            # print(score,baseline)
            score_scaled = (score - baseline)/(1-baseline)
            scores.append(score_scaled)
    return np.mean(scores)


'''
function:     tcap()   
description:  takes the original and synthetic dataset filenames and a set of keys/target variables and calculates the TCAP score
input:        original = location/filename of original dataset
              synth = location/filename of synthetic dataset
              num_keys = number of key variables
              target = target variable
              key = key variable as the baseline 
              verbose = if set to True it will print out more detailed results
output:       TCAP score and the baseline value for that target variable
'''
def tcap(orig, syn, target, key, verbose=False):
       
    # read in the data
    #orig = pd.read_csv(original)
    #syn = pd.read_csv(synth)
    
    # define the keys and target. using the num_keys parameter means that a dataset with any number of columns can
    # be used, and only the relevant keys analysed
    keys_target = key + [target]
    num_keys = len(key)
    # print(keys_target)
    
    # select just the required columns (keys and target)    
    orig = orig[keys_target]
    syn = syn[keys_target]
    # replace any missing values
    orig = replace_missing(orig)
    syn = replace_missing(syn)
    # count the categories for the target (for calculating baseline)
    uvd = orig[target].value_counts()
    
    # use groupby to get the equivalance classes for synthetic data
    eqkt_syn = pd.DataFrame({'count' : syn.groupby( keys_target ).size()}).reset_index()           # with target
    eqk_syn = pd.DataFrame({'count' : syn.groupby( keys_target[:num_keys] ).size()}).reset_index() # without target
    # equivalance classes for original data without target
    eqk_orig = pd.DataFrame({'count' : orig.groupby( keys_target[:num_keys] ).size()}).reset_index()

    # merge with original to calculate baseline    
    orig_merge_eqk = pd.merge(orig, eqk_orig, on= keys_target[:num_keys]) 
    orig_merge_eqk.rename({'count': 'count_eqk_orig'}, axis=1, inplace=True)
    # calculate the baseline
    uvt = sum(uvd[orig_merge_eqk[target]]/sum(uvd))
    baseline = uvt/len(orig)
    
    # calculate synthetic cap score. merge syn eq classes (with keys) with syn eq classes (with keys/target)
    syn_merge = eqk_syn.merge(eqkt_syn, on=keys_target[:num_keys])
    syn_merge['prop'] = syn_merge['count_y']/syn_merge['count_x']
    # filter out those less than tau=1
    syn_merge = syn_merge[syn_merge['prop'] >= 1]
    # merge with original, if in syn eq classes (just keys) then this is a matching record (Taub)
    syn_merge = syn_merge.merge(orig_merge_eqk, on=keys_target[:num_keys], how='inner')
    matching_records = len(syn_merge)

    # drop records where the targets are not equal
    syn_merge = syn_merge[syn_merge[target + '_x']==syn_merge[target + '_y']]
    dcaptotal = len(syn_merge)

    if matching_records == 0:
        tcap_undef = 0
    else:
        tcap_undef = dcaptotal/matching_records
   
    # output is [the TCAP as used by Taub, and the baseline]. Modify as required
    output = ([tcap_undef,baseline])
    
    if verbose==True:
        print('TCAP calculation')
        print('===============')
        print('Source dataset is: ',orig)
        print('Target dataset is: ',syn)
        print('The total number of records in the source dataset is: ', len(orig))
        print('The total number of records in the target dataset is: ', len(syn))
        print('The target variable is: ', target)
        print('The key size is: ', num_keys)
        print('The keys are: ', key)
        print('Number of matching records: ', matching_records)
        print('DCAP total is: ', dcaptotal)
        print('TCAP with non-matches undefined is: ', tcap_undef)
        print('The baseline is: ', baseline)

    return(output)

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
    real_cat_data = real_cat_data.replace('nan', np.nan, inplace=True)
    
    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    # new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    syn_cat_data = syn_cat_data.replace('nan', np.nan, inplace=True)
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    # new_syn_data.columns = range(len(new_syn_data.columns))
    
    metadata = info['metadata']

    # columns = metadata['columns']
    metadata['columns'] = {}

    # inverse_idx_mapping = info['inverse_idx_mapping']


    # for i in range(len(new_real_data.columns)):
    #     if i < len(num_col_idx):
    #         metadata['columns'][i] = columns[num_col_idx[i]]
    #     else:
    #         metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]

    return new_real_data, new_syn_data, metadata

#### Addition for WD, MMD, and KS Test

from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances

from scipy.stats import ks_2samp, chi2_contingency
from typing import List, Dict, Literal

import torch
import ot

def wasserstein_continuous(real_df, syn_df, cont_cols):
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_df[cont_cols])
    syn_scaled = scaler.transform(syn_df[cont_cols])

    wd_per_col = {}
    wd_clip_per_col = {}
    for i, col in enumerate(cont_cols):
        wd_per_col[col] = wasserstein_distance(
            real_scaled[:, i],
            syn_scaled[:, i]
        )
        wd_clip_per_col[col] = wasserstein_distance(
            real_scaled[:, i],
            syn_scaled[:, i].clip(0, 1)
        )

    return {
        "wd_mean": float(np.mean(list(wd_per_col.values()))),
        "wd_clip_mean": float(np.mean(list(wd_clip_per_col.values()))),
        "wd_max": float(np.max(list(wd_per_col.values()))),
        "wd_full": wd_per_col
    }


def mmd_rbf_torch(X: torch.Tensor, Y: torch.Tensor, sigmas: List[float] = [1.0],
                    device = torch.device("cuda")) -> float:
    """
    Core function MMD RBF menggunakan PyTorch.
    Input X, Y sudah harus Tensor di device yang benar.
    """
    # Pastikan ukuran sama (kalau batching, ini biasanya aman)
    n = X.shape[0]
    m = Y.shape[0]
    
    # Hitung pairwise distance squared Euclidean: ||x-y||^2
    # a^2 + b^2 - 2ab
    xx = torch.mm(X, X.t())
    yy = torch.mm(Y, Y.t())
    xy = torch.mm(X, Y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy

    mmd = 0.0
    for s in sigmas:
        gamma = 1.0 / (2 * s * s)
        
        # Kernel matrices
        Kxx = torch.exp(-gamma * dxx)
        Kyy = torch.exp(-gamma * dyy)
        Kxy = torch.exp(-gamma * dxy)
        
        # Zero diagonal for unbiased estimate (optional, sesuai kodemu yang lama)
        # Kalau mau unbiased, diagonal Kxx dan Kyy dinolkan.
        # Kalau biased V-statistic, biarkan diagonalnya 1.
        # Kode aslimu menolkan diagonal (Unbiased U-statistic-ish)
        Kxx = Kxx * (1 - torch.eye(n, device=device))
        Kyy = Kyy * (1 - torch.eye(m, device=device))
        
        # Rumus Unbiased Estimator
        # term1 = Kxx.sum() / (n * (n - 1))
        # term2 = Kyy.sum() / (m * (m - 1))
        # term3 = 2 * Kxy.mean()  # Kxy tidak perlu nol diagonal karena X != Y
        
        # Note: Kxy.mean() membagi dengan (n*m), sedangkan unbiased biasanya 2 * sum / (n*m)
        # Biar konsisten sama kode aslimu:
        
        term1 = Kxx.sum() / (n * (n - 1))
        term2 = Kyy.sum() / (m * (m - 1))
        term3 = 2 * Kxy.mean() 
        
        mmd += (term1 + term2 - term3)

    return mmd / len(sigmas)

def mmd_mixed_sliced_torch(
    real_df: pd.DataFrame, 
    syn_df: pd.DataFrame, 
    cont_cols: List[str], 
    cat_cols: List[str], 
    n_slices: int = 20,  # Misal 100rb / 20 = 5rb per slice
    sigmas: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0],
    device = torch.device("cuda")
) -> dict:
    
    # 1. Preprocessing (Sama seperti sebelumnya - Gabung, Scale, OHE)
    real_proc = real_df.copy()
    syn_proc = syn_df.copy()
    n_real = len(real_proc)
    
    combined = pd.concat([real_proc, syn_proc], axis=0, ignore_index=True)

    if cont_cols:
        scaler = MinMaxScaler()
        scaler.fit(real_proc[cont_cols])
        combined[cont_cols] = scaler.transform(combined[cont_cols])
    
    if cat_cols:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_encoded = enc.fit_transform(combined[cat_cols])
        cat_feat_names = enc.get_feature_names_out(cat_cols)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feat_names, index=combined.index)
        combined = combined.drop(columns=cat_cols)
        combined = pd.concat([combined, cat_df], axis=1)

    # Convert ke Numpy float32
    data_real_final = combined.iloc[:n_real].values.astype(np.float32)
    data_syn_final = combined.iloc[n_real:].values.astype(np.float32)

    # 2. SHUFFLE & SLICING
    # Kita acak urutan barisnya dulu biar slicenya random
    np.random.shuffle(data_real_final)
    np.random.shuffle(data_syn_final)

    # Hitung ukuran per slice
    # Kita ambil jumlah minimum biar adil (misal real 100k, syn 105k -> pake 100k)
    min_len = min(len(data_real_final), len(data_syn_final))
    slice_size = min_len // n_slices
    
    print(f"Total Data: {min_len}. Split into {n_slices} slices of size {slice_size}.")
    
    mmd_scores = []

    # 3. Loop per Slice (One-to-One Comparison)
    for i in range(n_slices):
        start_idx = i * slice_size
        end_idx = start_idx + slice_size
        
        # Ambil slice ke-i dari Real dan Syn
        slice_real = torch.tensor(data_real_final[start_idx:end_idx], device=device)
        slice_syn = torch.tensor(data_syn_final[start_idx:end_idx], device=device)
        
        # Hitung MMD
        score = mmd_rbf_torch(slice_real, slice_syn, sigmas=sigmas)
        
        mmd_scores.append(score.item())
        
        # Bersihkan cache GPU
        if device.type == 'cuda':
            del slice_real, slice_syn
            torch.cuda.empty_cache()

        print(f"Slice {i+1}/{n_slices}: MMD = {score.item():.6f}")

    # 4. Agregasi
    mean_mmd = np.mean(mmd_scores)
    std_mmd = np.std(mmd_scores)
    
    return {
        "mmd_mean": float(mean_mmd),
        "mmd_std": float(std_mmd),
        "n_slices": n_slices,
        "slice_size": slice_size
    }

# --- CONTOH CARA PAKAI ---
# real_df = pd.read_csv("data_asli.csv")
# syn_df = pd.read_csv("data_palsu.csv")

# cont_columns = ["age", "income", "balance"]
# cat_columns = ["gender", "city", "job"]

# result = mmd_mixed_batched_torch(
#     real_df, 
#     syn_df, 
#     cont_cols=cont_columns, 
#     cat_cols=cat_columns, 
#     batch_size=5000,   # 5rb per hitungan
#     n_batches=20       # Diulang 20 kali (total cover 100rb baris secara statistik)
# )

# print(result)

# --- HELPERS PREPROCESSING ---
def _build_shared_category_maps(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict]:
    maps = {}
    for col in df1.columns:
        cats = pd.Index(df1[col].dropna().unique()).union(pd.Index(df2[col].dropna().unique()))
        maps[col] = {cat: i for i, cat in enumerate(cats)}
    return maps

def _encode_dataframe(df: pd.DataFrame, maps: Dict[str, Dict]) -> np.ndarray:
    df_encoded = df.copy()
    for col, mapping in maps.items():
        df_encoded[col] = df[col].map(mapping).fillna(-1).astype(int)
    return df_encoded.values

# --- GPU COST FUNCTIONS ---
def pairwise_hamming(X_cat: torch.Tensor, Y_cat: torch.Tensor) -> torch.Tensor:
    """
    Hitung Hamming distance (proporsi beda) antar kategori.
    Output range: [0, 1]
    """
    # (N, 1, D) != (1, M, D) -> (N, M, D) bool
    diffs = (X_cat[:, None, :] != Y_cat[None, :, :]).float()
    return diffs.mean(dim=2)

def pairwise_euclidean(X_cont: torch.Tensor, Y_cont: torch.Tensor) -> torch.Tensor:
    """
    Hitung Euclidean distance antar data kontinu.
    Input diasumsikan sudah dinormalisasi (MinMax), jadi range kira-kira [0, sqrt(D)]
    Kita bisa normalisasi cost ini ke [0, 1] jika mau, atau biarkan L2 murni.
    """
    # cd = cdist(X, Y)
    return torch.cdist(X_cont, Y_cont, p=2.0)

# --- MAIN FUNCTION JOINT ---
def wasserstein_joint_sliced_torch(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    cont_cols: List[str],
    cat_cols: List[str],
    n_slices: int = 20,
    solver: Literal["sinkhorn", "emd"] = "sinkhorn",
    reg: float = 1e-2,
    alpha: float = 0.5,
    device = torch.device("cuda")  # Bobot penyeimbang: Cost = alpha * Cont + (1-alpha) * Cat
) -> dict:

    print("--- Starting Joint Wasserstein Calculation ---")
    
    # 1. PREPROCESSING (Gabungan biar konsisten)
    real_proc = real_df.copy()
    syn_proc = syn_df.copy()
    n_real = len(real_proc)
    
    # A. Continuous: MinMax Scaling [0, 1]
    # Penting agar Euclidean distance tidak mendominasi Hamming (yg max 1.0)
    if cont_cols:
        scaler = MinMaxScaler()
        # Fit pada gabungan data
        # all_cont = pd.concat([real_proc[cont_cols], syn_proc[cont_cols]], axis=0)
        scaler.fit(real_proc[cont_cols])
        
        real_cont_np = scaler.transform(real_proc[cont_cols]).astype(np.float32)
        syn_cont_np = scaler.transform(syn_proc[cont_cols]).astype(np.float32)
    else:
        real_cont_np = syn_cont_np = None

    # B. Categorical: Integer Encoding
    if cat_cols:
        maps = _build_shared_category_maps(real_proc[cat_cols], syn_proc[cat_cols])
        real_cat_np = _encode_dataframe(real_proc[cat_cols], maps).astype(np.int64) # Long for index
        syn_cat_np = _encode_dataframe(syn_proc[cat_cols], maps).astype(np.int64)
    else:
        real_cat_np = syn_cat_np = None

    # 2. SHUFFLE INDICES
    # Kita shuffle index saja biar cont dan cat tetap sinkron barisnya
    indices_real = np.arange(n_real)
    indices_syn = np.arange(len(syn_proc))
    
    np.random.shuffle(indices_real)
    np.random.shuffle(indices_syn)
    
    min_len = min(len(indices_real), len(indices_syn))
    slice_size = min_len // n_slices
    
    print(f"Data: {min_len} rows. Slices: {n_slices} (size {slice_size}). Alpha: {alpha}")
    
    wd_scores = []
    
    # 3. LOOP SLICING (GPU)
    for i in range(n_slices):
        start_idx = i * slice_size
        end_idx = start_idx + slice_size
        
        idx_r = indices_real[start_idx:end_idx]
        idx_s = indices_syn[start_idx:end_idx]
        
        cost_matrix = 0.0
        valid_comp = False # Flag buat cek minimal ada 1 jenis data

        # --- Hitung Cost Continuous ---
        if cont_cols:
            X_cont = torch.tensor(real_cont_np[idx_r], device=device)
            Y_cont = torch.tensor(syn_cont_np[idx_s], device=device)
            
            # Euclidean Distance Matrix (N x M)
            C_cont = pairwise_euclidean(X_cont, Y_cont)
            
            # Normalisasi opsional: Bagi dengan akar(jumlah fitur) agar range ~ [0, 1]
            # C_cont = C_cont / np.sqrt(len(cont_cols)) 
            
            cost_matrix += alpha * C_cont
            valid_comp = True
            
            del X_cont, Y_cont, C_cont

        # --- Hitung Cost Categorical ---
        if cat_cols:
            X_cat = torch.tensor(real_cat_np[idx_r], device=device)
            Y_cat = torch.tensor(syn_cat_np[idx_s], device=device)
            
            # Hamming Distance Matrix (N x M) -> values [0, 1]
            C_cat = pairwise_hamming(X_cat, Y_cat)
            
            # Jika ada cont, bobotnya (1-alpha). Jika cuma cat, bobot 1.
            weight = (1.0 - alpha) if cont_cols else 1.0
            
            cost_matrix += weight * C_cat
            valid_comp = True
            
            del X_cat, Y_cat, C_cat
            
        if not valid_comp:
            raise ValueError("No columns provided (cont_cols and cat_cols are both empty)!")

        # --- Optimal Transport ---
        n = slice_size # asumsi potong rata
        a = torch.ones(n, device=device) / n
        b = torch.ones(n, device=device) / n
        
        try:
            if solver == "sinkhorn":
                loss = ot.sinkhorn2(a, b, cost_matrix, reg=reg)
            else:
                loss = ot.emd2(a, b, cost_matrix)
            
            wd_scores.append(loss.item())
        except Exception as e:
            print(f"Slice {i} error: {e}")
            wd_scores.append(np.nan)
        
        # Cleanup
        del cost_matrix, a, b
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        print(f"Slice {i+1}/{n_slices}: Joint WD = {wd_scores[-1]:.6f}")

    # 4. AGGREGATE
    return {
        "wd_mean": float(np.nanmean(wd_scores)),
        "wd_std": float(np.nanstd(wd_scores)),
        "params": {"alpha": alpha, "solver": solver}
    }

# --- CONTOH PAKAI ---
# result = wasserstein_joint_sliced_torch(
#    real_df, syn_df,
#    cont_cols=["age", "balance"],
#    cat_cols=["job", "marital"],
#    n_slices=20,
#    alpha=0.5  # 50% Bobot Continuous, 50% Bobot Categorical
# )


def ks_tests(real_df, syn_df, cont_cols):
    return {
        col: ks_2samp(real_df[col], syn_df[col]).pvalue
        for col in cont_cols
    }

def chi_square_tests(real_df, syn_df, cat_cols):
    results = {}
    for col in cat_cols:
        tbl = pd.concat([
            real_df[col].value_counts(),
            syn_df[col].value_counts()
        ], axis=1).fillna(0)
        _, pval, _, _ = chi2_contingency(tbl.values)
        results[col] = pval
    return results

def aggregate_pvalues(pvals, alpha=0.05):
    pvals_arr = np.array(list(pvals.values()))
    return {
        "mean_p": float(pvals_arr.mean()),
        "frac_reject": float((pvals_arr < alpha).mean())
    }


def ks_similarity(real_df, syn_df, cont_cols):
    ks_scores = {}
    pvals = {}

    for col in cont_cols:
        D, p = ks_2samp(real_df[col], syn_df[col])
        ks_scores[col] = 1.0 - D   # SDV-style
        pvals[col] = p

    return {
        "ks_score_mean": float(np.mean(list(ks_scores.values()))),
        "ks_score_min": float(np.min(list(ks_scores.values()))),
        "pval_mean": float(np.mean(list(pvals.values()))),
        "frac_reject": float(np.mean(np.array(list(pvals.values())) < 0.05))
    }

def tvd_categorical(real_df, syn_df, cat_cols):
    tvds = {}
    for col in cat_cols:
        p = real_df[col].value_counts(normalize=True)
        q = syn_df[col].value_counts(normalize=True)
        idx = p.index.union(q.index)
        p = p.reindex(idx, fill_value=0)
        q = q.reindex(idx, fill_value=0)
        tvds[col] = 0.5 * np.abs(p - q).sum()
    return {
        "tvd_mean": float(np.mean(list(tvds.values()))),
        "tvd_max": float(np.max(list(tvds.values()))),
        "tvd_full": tvds
    }

def compute_pairwise_distance(data_x, data_y=None): # Changed to L1 instead of L2 to better handle mixed data
	"""
	Args:
		data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
		data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
	Returns:
		numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
	"""
	if data_y is None:
		data_y = data_x
	dists = pairwise_distances(
		data_x, data_y, metric='cityblock', n_jobs=-1)
	return dists


# def wasserstein_ot_gower(real_df, syn_df, cont_cols, cat_cols, reg=0.05):
#     scaler = MinMaxScaler()
#     scaler.fit(real_df[cont_cols])
#     # real_scaled = scaler.fit_transform(real_df[cont_cols])
#     # syn_scaled = scaler.transform(syn_df[cont_cols])

#     enc = make_ohe()
#     enc.fit(real_df[cat_cols])
    
#     real_features = np.concatenate((scaler.transform(real_df[cont_cols]), 
#                              0.5 * enc.transform(real_df[cat_cols])), axis = 1)
#     fake_features = np.concatenate((scaler.transform(syn_df[cont_cols]), 
#                              0.5 * enc.transform(syn_df[cat_cols])), axis = 1)
    
#     n = real_features.shape[0]
#     m = fake_features.shape[0]

#     a = np.ones(n) / n
#     b = np.ones(m) / m

#     C = compute_pairwise_distance(real_features, fake_features)
#     C = C / C.max()  # optional normalization

#     wd = ot.sinkhorn2(a, b, C, reg)
#     return float(wd)

# import torch
# from geomloss import SamplesLoss ## use "pip install geomloss[full]" if error in generic_logsumexp 

# def wasserstein_ot_gower(
#     real_df,
#     syn_df,
#     cont_cols,
#     cat_cols,
#     blur=0.05,
#     device="cuda"
# ):
#     # === preprocessing (SAMA seperti punyamu) ===
#     scaler = MinMaxScaler()
#     scaler.fit(real_df[cont_cols])

#     enc = make_ohe()
#     enc.fit(real_df[cat_cols])

#     real_features = np.concatenate((
#         scaler.transform(real_df[cont_cols]),
#         0.5 * enc.transform(real_df[cat_cols])
#     ), axis=1)

#     fake_features = np.concatenate((
#         scaler.transform(syn_df[cont_cols]),
#         0.5 * enc.transform(syn_df[cat_cols])
#     ), axis=1)

#     # === pindah ke torch ===
#     x = torch.tensor(real_features, dtype=torch.float32, device=device)
#     y = torch.tensor(fake_features, dtype=torch.float32, device=device)

#     # === Sinkhorn Wasserstein (implicit pairwise) ===
#     loss = SamplesLoss(
#         loss="sinkhorn",
#         p=1,        # L1 → paling dekat ke Gower
#         blur=blur   # regularization (≈ reg di POT)
#     )

#     wd = loss(x, y)
#     return float(wd.detach().cpu().item())



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

    syn_data = pd.read_csv(syn_path, dtype=str)
    real_data = pd.read_csv(real_path, dtype=str)

    discrete_columns = [real_data.columns[i] for i in info['cat_col_idx']]
    numerical_columns = [real_data.columns[i] for i in info['num_col_idx']]
    if info['task_type'] == 'binclass': discrete_columns += real_data.columns[info['target_col_idx']].tolist()
    else: numerical_columns += real_data.columns[info['target_col_idx']].tolist()

    # print(real_data.values)
    # print(real_data.info())

    # print(syn_data.head())
    # print(syn_data.values)
    
    real_data[numerical_columns] = real_data[numerical_columns].astype(float)
    if dataname not in ['adulta', 'churn','tcga','tcgaa','diabetes']:
        real_data['AGE'] = real_data['AGE'].round(0).astype(int)
    
    # real_data[discrete_columns] = real_data[discrete_columns].replace('nan', np.nan, inplace=True)
    real_data[discrete_columns] = real_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                      value=r'\1', regex=True)
    
    syn_data[numerical_columns] = syn_data[numerical_columns].astype(float)
    if dataname not in ['adulta', 'churn', 'tcga','tcgaa','diabetes']:
        syn_data['AGE'] = syn_data['AGE'].round(0).astype(int)
    # syn_data[discrete_columns] = syn_data[discrete_columns].replace('nan', np.nan, inplace=True)
    syn_data[discrete_columns] = syn_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                      value=r'\1', regex=True)
    
    if dataname == 'indonesia':
        cols_to_fix = ['LANDOWN','HOMEFEM','HOMEMALE']
        # syn_data[cols_to_fix] = syn_data[cols_to_fix].replace({0: "00", "0": "00"})
        syn_data[cols_to_fix] = syn_data[cols_to_fix].replace({str(j): f'0{j}' for j in range(10)})
        syn_data['EDATTAIND'] = syn_data['EDATTAIND'].replace({0: "000", "0": "000"})
    
    # print(real_data.head())
    # print(syn_data.head())
    # new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    print(real_data[discrete_columns].shape, syn_data[discrete_columns].shape)
    for i in range(real_data[discrete_columns].to_numpy().shape[1]):
        print(i, 'real', np.unique(real_data[discrete_columns].to_numpy()[:,i]))
        print(i, 'synt', np.unique(syn_data[discrete_columns].to_numpy()[:,i]))

    ## utility and risk
    results = eval_syn_data(dataname, real_data, syn_data)
    # if dataname not in ['tcga','tcgaa']:
    #     results = eval_syn_data(dataname, real_data, syn_data)
    # else:
    #     results = {'ROC_uni': 0.0, 'ROC_biv': 0.0, 'CIO': 0.0, 'TCAP': 0.0, 
    #                'Utility':0.0, 'Risk': 0.0}
    
    # ## Additional metrics
    # wd_gower = wasserstein_ot_gower(real_data, syn_data, numerical_columns, discrete_columns)
    # results["WD_gower"] = wd_gower
    
    # wd_scores = wasserstein_continuous(real_data, syn_data, numerical_columns)
    # results["WD"] = wd_scores['wd_mean']
    # results["WD_clip"] = wd_scores['wd_clip_mean']

    # wd_scores = mmd_mixed_sliced_torch(
    #     real_data, syn_data, 
    #     numerical_columns, discrete_columns,
    #     n_slices = (len(real_data) // 10000) + 1,  # 5rb per hitungan
    # )
    # results["WD"] = wd_scores['mmd_mean']

    # mmd_scores = wasserstein_joint_sliced_torch(
    #     real_data, syn_data, 
    #     numerical_columns, discrete_columns,
    #     n_slices = (len(real_data) // 10000) + 1,
    #     alpha=0.5  # 50% Bobot Continuous, 50% Bobot Categorical
    # )
    # results["MMD"] = mmd_scores['wd_mean']


    # mmd_scores = mmd_categorical(real_data, syn_data, discrete_columns, 10000)
    # results["MMD"] = mmd_scores['mmd_cat']

    # ks_raw = ks_tests(real_data, syn_data, numerical_columns)
    # chi2_raw = chi_square_tests(real_data, syn_data, discrete_columns)

    # ks_mean_sim = ks_similarity(real_data, syn_data, numerical_columns)
    
    # results["KS_sim"] = ks_mean_sim['ks_score_mean']
    # results["tvd_sim"] = tvd_categorical(real_data, syn_data, discrete_columns)['tvd_mean']

    # results["KS_pval"] = ks_mean_sim['pval_mean']
    # results["ChiSquare_pval"] = aggregate_pvalues(chi2_raw)['mean_p']

    # results["KS_frac_reject"] = ks_mean_sim['frac_reject']
    # results["ChiSquare_frac_reject"] = aggregate_pvalues(chi2_raw)['frac_reject']

    save_dir = f'eval/urisk/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'eval/urisk/{dataname}/{model}.json'
    print('Saving scores to ', save_path)

    with open(save_path, "w") as json_file:
        json.dump(results, json_file, indent=4, separators=(", ", ": "))
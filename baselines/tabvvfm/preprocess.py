from ctgan.data_transformer import DataTransformer
import itertools, numpy as np

def prepare_data_trans_ctgan(data,discrete_columns = None):
    discrete_columns = data.columns if discrete_columns is None else discrete_columns
    tran = DataTransformer()
    tran.fit(data,discrete_columns)
    ## needed attributes
    # colnames = data.columns
    data_info_for_tddpm = list(itertools.chain(*tran.output_info_list))
    return tran, data_info_for_tddpm

def prepare_ctgan_prepr_to_tddpm(data,tran,data_info_for_tddpm):
    tran_data = tran.transform(data)
    X_num = np.zeros((data.shape[0],1))
    X_cat = np.zeros((data.shape[0],1))
    X_cat_dummy = np.zeros((tran_data.shape[0],1))

    num_list = []
    cat_list = []

    st = 0
    for i in range(len(data_info_for_tddpm)):
        ed = st + data_info_for_tddpm[i].dim
#         print(st,ed,ed-st)
        if data_info_for_tddpm[i].activation_fn == 'tanh':
            X_num = np.concatenate((X_num,tran_data[:,st:ed]),axis=1)
            num_list.append(data_info_for_tddpm[i])
        elif data_info_for_tddpm[i].activation_fn == 'softmax':
            X_cat = np.concatenate([X_cat,np.argmax(tran_data[:,st:ed],axis=1,keepdims=True)],axis=1)
            X_cat_dummy = np.concatenate([X_cat_dummy,tran_data[:,st:ed]],axis=1)
            cat_list.append(data_info_for_tddpm[i])
        st = ed
    X_num = X_num[:,1:] if X_num.shape[1] > 1 else None
    X_cat = X_cat[:,1:] if X_cat.shape[1] > 1 else None
    X_cat_dummy = X_cat_dummy[:,1:] if X_cat.shape[1] > 1 else None
    return X_num, X_cat, X_cat_dummy, num_list, cat_list

def prepare_tddpm_to_ctgan_prepr(X_num,X_cat_dummy,tran,data_info_for_tddpm,num_list,cat_list):
    X_concat = np.zeros((X_cat_dummy.shape[0],1)) if X_num is None else np.zeros((X_num.shape[0],1))
    st_num = 0
    st_cat = 0
    count_num = 0
    count_cat = 0
    for i in range(len(data_info_for_tddpm)):
        if data_info_for_tddpm[i].activation_fn == 'tanh':
            ed_num = st_num + num_list[count_num].dim
            count_num += 1
            X_concat = np.concatenate([X_concat,X_num[:,st_num:ed_num]],axis=1)
            st_num = ed_num
        elif data_info_for_tddpm[i].activation_fn == 'softmax':
            ed_cat = st_cat + cat_list[count_cat].dim
            count_cat += 1
            X_concat = np.concatenate([X_concat,X_cat_dummy[:,st_cat:ed_cat]],axis=1)
            st_cat = ed_cat
    X_concat = X_concat[:,1:]
    # print(X_concat.shape)
    return tran.inverse_transform(X_concat)
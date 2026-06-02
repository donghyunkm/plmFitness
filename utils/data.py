import os
import pandas as pd
import torch
from collections import defaultdict
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def trunc_sequence(protein, max_len):
    L = len(protein['wild_type'])
    if L <= max_len:
        protein['offset'] = 0
        return
    
    df = protein['df']
    positions = list(chain(*df['positions']))
    max_pos, min_pos = max(positions), min(positions)
    gap = max_pos - min_pos + 1
    
    if max_pos < max_len:
        protein['wild_type'] = protein['wild_type'][:max_len]
        protein['offset'] = 0
        return
    
    if gap <= max_len:
        window_l = max(min_pos - (max_len - gap) // 2, 0)
        window_r = min(max_pos + (max_len - gap) // 2, L - 1)
        seq_lr = protein['wild_type'][window_l: window_l + max_len]
        seq_rl = protein['wild_type'][window_r - max_len + 1: window_r + 1]
        
        if len(seq_lr) > len(seq_rl):
            protein['wild_type'] = seq_lr
            left, right = window_l, window_l + max_len
        else:
            protein['wild_type'] = seq_rl
            left, right = window_r - max_len + 1, window_r + 1
    else:
        n = 0
        left, right = min_pos, max_len
        window_l, window_r = min_pos, max_len
        while window_r < L:
            window_n = df['positions'].apply(
                lambda positions: all(window_l <= pos < window_r for pos in positions)).sum()
            if window_n > n:
                left, right = window_l, window_r
                n = window_n
            window_l += 1
            window_r += 1
        
        if right - left + 1 < max_len:
            left = right - max_len
        protein['wild_type'] = protein['wild_type'][left:right]
    
    df_bool = df.apply(lambda row: all(left <= pos < right for pos in row['positions']), axis=1)
    df = df.loc[df_bool].copy()
    df.loc[:, 'positions'] = df['positions'].apply(lambda positions: tuple(pos - left for pos in positions))
    protein['df'] = df
    protein['offset'] = left
    return

def process_dms(file_path, shuffle=True, max_len=1022, wild_type=None):
    df = pd.read_csv(file_path, index_col='mutant')
    if shuffle:
        df = df.sample(frac=1)
   
    new_df, n_sites = defaultdict(list), set()
    for mutant, row in df.iterrows():
        wt_aas, mt_aas, positions = '', '', []
        for site in mutant.split(':'): # handle multi-site mutants
            wt_aa, position, mt_aa = site[0], int(site[1:-1]) - 1, site[-1]
            if wild_type is None:
                assert row['mutated_sequence'][position] == mt_aa
            else:
                assert wild_type[position] == wt_aa
            wt_aas += wt_aa
            mt_aas += mt_aa
            positions.append(position)
        
        new_df['wt_aas'].append(wt_aas)
        new_df['mt_aas'].append(mt_aas)
        new_df['positions'].append(tuple(positions))
        n_sites.add(len(positions))
    
    new_df = pd.concat([pd.DataFrame(new_df, index=df.index),
                        df[['DMS_score', 'DMS_score_bin']]], axis=1)
    if wild_type is None:
        wild_type = list(row['mutated_sequence'])
        for wt_aa, position in zip(wt_aas, positions): # recover wild type sequence
            wild_type[position] = wt_aa
        wild_type = ''.join(wild_type)
    protein = dict(wild_type=wild_type, df=new_df)
    trunc_sequence(protein, max_len)
    protein['n_sites'] = sorted(n_sites)
    protein['name'] = os.path.basename(file_path).split('.')[0]
    return protein

def merge_files(data_dir, shuffle=True, max_len=1022, save_path=None):
    file_names = os.listdir(data_dir)
    proteins = defaultdict(list)
    for file_name in file_names:
        if 'indels' in file_name:
            continue
        protein = process_dms(f'{data_dir}/{file_name}', shuffle, max_len)
        name = '_'.join(file_name.split('_')[:2])
        proteins[name].append(protein)
    
    if save_path is not None:
        make_dir(save_path)
        torch.save(proteins, save_path)
    return proteins

def normalize(train_df, *eval_dfs):
    train_scores = train_df['DMS_score'].to_numpy()[:,None]
    scaler = StandardScaler()
    train_df['DMS_score'] = scaler.fit_transform(train_scores).squeeze(1)
    for eval_df in eval_dfs:
        if not eval_df.empty:
            eval_scores = eval_df['DMS_score'].to_numpy()[:,None]
            eval_df['DMS_score'] = scaler.transform(eval_scores).squeeze(1)

def _stratify_labels(df):
    if 'DMS_score_bin' not in df:
        raise ValueError('Cannot stratify split because DMS_score_bin is missing.')
    return df['DMS_score_bin']

def _split_ids(df, **kwargs):
    return train_test_split(
        df.index,
        stratify=_stratify_labels(df),
        **kwargs,
    )

def split_data(protein, train_size=0.8, shuffle=False, n_sites=None, neg_train=False,
               scale=False, train_ids=None, random_state=0, test_size=0.2,
               return_valid=False):
    df = protein['df']
    train, test = protein.copy(), protein.copy()
    valid = protein.copy()
    valid['df'] = df.iloc[:0]
    
    if train_ids is not None:
        train['df'] = df.loc[train_ids]
        test['df'] = df.loc[df.index.difference(train_ids, sort=False)]
    else:
        if shuffle:
            df = df.sample(frac=1, random_state=random_state)
        if n_sites is not None:
            n_sites = set(n_sites)

        if test_size:
            train_pool_ids, test_ids = _split_ids(
                df,
                test_size=test_size,
                random_state=random_state,
            )
            train_pool_df = df.loc[train_pool_ids]
            test['df'] = df.loc[test_ids]
        else:
            train_pool_df = df
            test['df'] = df.iloc[:0]
    
        df_bool = train_pool_df.apply(lambda row: (not n_sites or len(row['positions']) in n_sites) and \
                                                 (not neg_train or row['DMS_score_bin'] == 0), axis=1)
        eligible_df = train_pool_df.loc[df_bool]
        if train_size <= 1:
            train_size = int(len(eligible_df) * train_size)
        else:
            train_size = min(int(train_size), len(eligible_df))
        if train_size <= 0:
            subset_df = eligible_df.iloc[:0]
        elif train_size < len(eligible_df):
            subset_ids, _ = _split_ids(
                eligible_df,
                train_size=train_size,
                random_state=random_state,
            )
            subset_df = eligible_df.loc[subset_ids]
        else:
            subset_df = eligible_df

        if return_valid and len(subset_df) > 0:
            train_ids, valid_ids = _split_ids(
                subset_df,
                test_size=0.2,
                random_state=random_state,
            )
            train['df'] = subset_df.loc[train_ids]
            valid['df'] = subset_df.loc[valid_ids]
        else:
            train['df'] = subset_df
    
    if scale:
        normalize(train['df'], valid['df'], test['df'])
    if return_valid:
        return train, valid, test
    return train, test

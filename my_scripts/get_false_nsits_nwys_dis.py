#!/usr/bin/env python3
import pandas as pd
import json
from collections import Counter
from get_nsites_nwys_dis import get_nsites_dis, get_nwys_dis

def get_false_structure_idx():
    false_idxs = []
    for i in range(1, 231):
        js_path = f'eval_metrics_{i}.json'
        js = json.load(open(js_path, "r"))
        lst = js["validity"]["each_valid"]
        false_idx = [i for i,v in enumerate(lst) if not v]
        false_idxs.append(false_idx)
    return false_idxs

def get_nwys_nsites_counter_from_idx(false_idxs):
    false_dfs = []
    for i in range(230):
        file_path = f"output_{i+1}.csv"
        df = pd.read_csv(file_path)
        false_df = df.iloc[false_idxs[i]]
        false_dfs.append(false_df)
    false_df_all = pd.concat(false_dfs)

    nwys_key, nwys_value, nwys_df = get_nwys_dis(false_df_all)
    nsites_key, nsites_value, nsites_df = get_nsites_dis(false_df_all)
    
    return nwys_key, nwys_value, nwys_df, nsites_key, nsites_value, nsites_df

def main():
    false_idxs = get_false_structure_idx()
    nwys_key, nwys_value, nwys_df, nsites_key, nsites_value, nsites_df = get_nwys_nsites_counter_from_idx(false_idxs)
    print(f'wyckoff number distribution: {dict(zip(nwys_key, nwys_value))}')
    print(f'atoms number distribution: {dict(zip(nsites_key, nsites_value))}')
    nwys_df.to_csv("false_nwys_dis.csv",index=False)
    nsites_df.to_csv("false_nsites_dis.csv",index=False)
    false_num = []
    for i in false_idxs:
        false_num += i
    print(f"total false structure number is {false_num}")

main()

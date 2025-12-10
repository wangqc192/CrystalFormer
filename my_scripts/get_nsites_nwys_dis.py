#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from collections import Counter
from ast import literal_eval

def get_nsites_dis(df_all):
    nsites = df_all["M"].apply(literal_eval).apply(sum)
    counter_nsites = Counter(nsites)
    key, value = zip(*sorted(counter_nsites.items()))
    df = pd.DataFrame({'nsites':key, 'counter':value})
    return key, value, df

def get_nwys_dis(df_all):
    nwys = df_all["W"].apply(literal_eval).apply(lambda w: sum(1 for v in w if v != 0))
    counter_nwys = Counter(nwys)
    key, value = zip(*sorted(counter_nwys.items()))
    df = pd.DataFrame({'nwys':key, 'counter':value})
    return key, value, df

def main():
    file_path = Path("./")
    dfs = []
    for i in file_path.glob("output_*.csv"):
        if "struct" not in str(i):
            df = pd.read_csv(i)
            dfs.append(df)
    df_all = pd.concat(dfs)

    k_s, v_s, df_nsites = get_nsites_dis(df_all)
    k_w, v_w, df_nwys = get_nwys_dis(df_all)
    print(f'atoms number distribution: {dict(zip(k_s, v_s))}')
    print(f'wyckoff number distribution: {dict(zip(k_w, v_w))}')
    df_nsites.to_csv("nsites_dis.csv",index=False)
    df_nwys.to_csv("nwys_dis.csv",index=False)

if __name__ == "__main__":
    main()

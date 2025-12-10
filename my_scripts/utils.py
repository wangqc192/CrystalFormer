import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee", "no-latex"])


def split_data(keys, values, threshold):
    if threshold is None:
        return keys, values, [], []   # 不切分

    main_keys = []
    main_vals = []
    tail_keys = []
    tail_vals = []

    for k, v in zip(keys, values):
        if k <= threshold:
            main_keys.append(k)
            main_vals.append(v)
        else:
            tail_keys.append(k)
            tail_vals.append(v)
    return main_keys, main_vals, tail_keys, tail_vals

import numpy as np
def plot(
        datasets,
        threshold=None,
        xlabel="value",
        ylabel="count",
        title='distribution',
        outfile="output.png",
        figsize=(16,6),
        ax_in_width = "45%",
        ax_in_height = "45%",
        xticks = None,
        ):

    split_results = []
    for keys, values, label in datasets:
        main_k, main_v, tail_k, tail_v = split_data(keys, values, threshold)
        split_results.append((label, main_k, main_v, tail_k, tail_v))
    
    fig, ax = plt.subplots(figsize=figsize)
    N = len(datasets)
    w = 0.8 / N
    offsets = np.linspace(-0.4 + w/2, 0.4 - w/2, N)

    for (label, main_k, main_v, tail_k, tail_v), offset in zip(split_results, offsets):
        ax.bar([k + offset for k in main_k], main_v, width=w, label=label)
    
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=30)

    # 横坐标旋转防止重叠
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    if xticks is not None:
         ax.set_xticks(xticks)
    
    # -----------------------------
    # 2. 添加嵌入子图 (Inset Axes)
    # -----------------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # inset_axes 参数位置和大小可以调整
    if threshold is not None:
        ax_in = inset_axes(ax, width=ax_in_width, height=ax_in_height, loc='upper right')
    
        for (label, main_k, main_v, tail_k, tail_v), offset in zip(split_results, offsets):
                    ax_in.bar([k + offset for k in tail_k], tail_v, width=w, label=label) 
    
        ax_in.set_title(f"{xlabel} > {threshold}", fontsize=18)
        ax_in.tick_params(axis="x", rotation=45, labelsize=8)
        ax_in.tick_params(axis="y", labelsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)



def _get_nwys(ser):
    for i in range(20):
        wp_i = int(ser[f'wp{i}'])
        if i != 19:
            wp_i_add_1 = int(ser[f'wp{i+1}'])
            if wp_i != -1 and wp_i_add_1 == -1:
                nwys = i+1
                return nwys
        else:
            if wp_i != -1:
                nwys = i+1
            else:
                nwys = i
            return nwys
        
def get_nwys_from_df(df):
    nwyss = []
    for i in range(len(df)):
        nwys = _get_nwys(df.iloc[i])
        nwyss.append(nwys)
    return nwyss
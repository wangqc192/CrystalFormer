#!/usr/bin/env python3
from structure_formation_transform import convert_wyckoff_symbol_number, convert_element_symbol_number
import pandas as pd
import json
import sys

def convert_former_out(df, spg):
    temp = []
    for i in range(len(df)):
        eles = eval(df["A"].iloc[i]) 
        wys = eval(df["W"].iloc[i])

        eles = [convert_element_symbol_number(j) for j in eles if j!= 0 ]
        wys = [convert_wyckoff_symbol_number(j) for j in wys if j!= 0]
        wys = "".join(wys)
        temp_i = {
            "spacegroup_number": spg,
            "wyckoff_letters": wys,
            "atom_types": eles}
        temp.append(temp_i)
    with open(f"temp_{spg}.json", 'w') as f:
        json.dump(temp, f)

csv = sys.argv[1]
def main():
    df = pd.read_csv(csv)
    spg = int(csv.split("_")[1].split(".")[0])
    convert_former_out(df,spg)
    

    
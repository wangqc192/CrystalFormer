import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import pickle

from crystalformer.src.wyckoff import mult_table
from crystalformer.src.elements import element_list

from pyxtal.lattice import Lattice
from pyxtal.wyckoff_site import Wyckoff_position
import ast
import pyxtal.symmetry as sym
from crystalformer.src.wyckoff import ss_mapping, ss_idx_mapping, ss_num_mapping

@jax.vmap
def sort_atoms(W, A, X):
    """
    lex sort atoms according W, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim) int
    """
    W_temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort

    X -= jnp.floor(X)
    idx = jnp.lexsort((X[:,2], X[:,1], X[:,0], W_temp))

    #assert jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]
    return A, X

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(key, data):
    """
    shuffle data along batch dimension
    """
    G, L, XYZ, A, W, S, I = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx], S[idx], I[idx]

def row_to_pyxtal(row):
    spg = int(row["spg"])
    a, b, c = row["a"], row["b"], row["c"]
    alpha, beta, gamma = row["alpha"] * 180/np.pi, row["beta"] * 180 / np.pi, row["gamma"] * 180/np.pi

    spe = ast.literal_eval(row["spe"])  # 例如 ['Li', 'Mn', 'Ir', 'Ir']

    xtal = pyxtal()
    lattice = Lattice.from_para(a, b, c, alpha, beta, gamma, radians=True)
    sites = []
    numIons = []

    for i in range(20):
        wp_index = int(row[f"wp{i}"])
        x = float(row[f"x{i}"])
        y = float(row[f"y{i}"])
        z = float(row[f"z{i}"])
        if wp_index != -1:
            wp = sym.Group(spg)[wp_index]
            multi = wp.multiplicity
            wyckoff_symbol = f'{multi}{wp.letter}'
            site = [{f"{wyckoff_symbol}": [x,y,z]}]
            #print(site)
            sites.append(site)
            numIons.append(multi)

        else:
            continue

    xtal.build(spg,spe,numIons,lattice,sites)

    return xtal
    
def process_one(row, atom_types, wyck_types, n_max, tol=0.01, is_cif=True):
    """
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    Process one cif string to get G, L, XYZ, A, W

    Args:
      cif: cif string
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      tol: tolerance for pyxtal

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    if "cif" in row.columns:
        cif = row["cif"]
        try: crystal = Structure.from_str(cif, fmt='cif')
        except: crystal = Structure.from_dict(eval(cif))
        spga = SpacegroupAnalyzer(crystal, symprec=tol)
        crystal = spga.get_refined_structure()
        c = pyxtal()
        try:
            c.from_seed(crystal, tol=0.01)
        except:
            c.from_seed(crystal, tol=0.0001)
    else:
        c = row_to_pyxtal(row)

    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) # we will need at least one empty site for output of L params

    #print (g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    ss = []
    ii = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        s = ss_mapping[g-1][site.wp.letter]
        s = ss_num_mapping[s]
        i = ss_idx_mapping[g-1][site.wp.letter]
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        ss.append(s)
        ii.append(i)
        #print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    ss = np.array(ss)[idx]
    ii = np.array(ii)[idx]
    #print (ws, aa, ww, natoms) 

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)

    ss = np.concatenate([ss,
                        np.full((n_max - num_sites, ), -1)],
                        axis=0)
    
    ii = np.concatenate([ii,
                        np.full((n_max - num_sites, ), -1)],
                        axis=0)

    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    #print ('===================================')

    return g, l, fc, aa, ww, ss, ii 

def GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, num_workers=1, is_cif=True):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W
    Note that cif strings must be in the column 'cif'

    Args:
      csv_file: csv file containing cif strings
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      num_workers: number of workers for multiprocessing

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    print(f"Load {csv_file}")
    data = pd.read_csv(csv_file)
    print(f"Loaded {csv_file}")

    print("Start processing data")
    results = Parallel(num_workers, backend="multiprocessing")(
            delayed(process_one)(
                data.iloc[i], atom_types=atom_types, wyck_types=wyck_types, n_max=n_max, is_cif=is_cif
                ) 
            for i in tqdm(range(len(data))
                    )
                )

    G, L, XYZ, A, W, S, I = zip(*results)

    G = jnp.array(G) 
    A = jnp.array(A).reshape(-1, n_max)
    W = jnp.array(W).reshape(-1, n_max)
    XYZ = jnp.array(XYZ).reshape(-1, n_max, 3)
    L = jnp.array(L).reshape(-1, 6)
    S = jnp.array(S).reshape(-1, n_max)
    I = jnp.array(I).reshape(-1, n_max)

    A, XYZ = sort_atoms(W, A, XYZ)

    save_path = os.path.splitext(csv_file)[0] + ".pt"
    pickle.dump((G,L,XYZ,A,W,S,I), open(save_path, "wb"))
    
    return G, L, XYZ, A, W, S, I

def GLXA_to_structure_single(G, L, X, A):
    """
    Convert G, L, X, A to pymatgen structure. Do not use this function due to the bug in pymatgen.

    Args:
      G: space group number
      L: lattice parameters
      X: fractional coordinates
      A: atom types
    
    Returns:
      structure: pymatgen structure
    """
    lattice = Lattice.from_parameters(*L)
    # filter out padding atoms
    idx = np.where(A > 0)
    A = A[idx]
    X = X[idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    if isinstance(G, int):
        G = np.array([G] * len(L))
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    atom_types = 119
    wyck_types = 28
    n_max = 21

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    #csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    csv_file = '/root/autodl-tmp/CrystalFormer/data/mp_20_aug_V3/train.csv'

    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)
    
#    print (G.shape)
#    print (L.shape)
#    print (XYZ.shape)
#    print (A.shape)
#    print (W.shape)
    
#    print ('L:\n',L)
#    print ('XYZ:\n',XYZ)


    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print ('N:\n', M.sum(axis=-1))

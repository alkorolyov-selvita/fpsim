from tqdm import trange
import numpy as np
from time import time
import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from rdkit.DataStructs.cDataStructs import ExplicitBitVect, BulkTanimotoSimilarity
from chemivec import (tanimoto_similarity_matrix_square,
                            tanimoto_similarity_matrix,
                            calc_cross_rmsd)
# from bitset.bitset import Bitset

def measure_runtime(func, *args, n_runs=7):
    """Measure the runtime of a function over multiple runs."""
    run_times = []
    for _ in trange(n_runs):
        tic = time()
        result = func(*args)
        toc = time()
        run_times.append(toc - tic)
    run_times = np.array(run_times)
    print(f'{n_runs} runs {run_times.mean():.3f} ± {run_times.std():.3f} s')
    return result

def calc_rmsd_np(arr, ref_arr):
    """Calculate RMSD between two arrays."""
    return np.sqrt(np.mean((arr[:, np.newaxis] - ref_arr[np.newaxis, :])**2, axis=1))

def jaccard_matrix_symm_rdkit(arr):
    if isinstance(arr, pd.Series):
        arr = arr.values
    res = np.zeros((len(arr), len(arr)), dtype=np.float32)
    for i in range(len(arr)):
        res[i] = np.array(BulkTanimotoSimilarity(arr[i], arr))
    return res

def jaccard_matrix_asymm_rdkit(arr1, arr2):
    if isinstance(arr1, pd.Series):
        arr1 = arr1.values
    if isinstance(arr2, pd.Series):
        arr2 = arr2.values

    res = np.zeros((len(arr1), len(arr2)), dtype=np.float32)
    for i in range(len(arr1)):
        res[i] = np.array(BulkTanimotoSimilarity(arr1[i], arr2))
    return res

if __name__ == '__main__':

    df = pd.read_pickle('/home/ergot/projects/tadam-pipeline/data/processed/chembl_activities.pkl.zst')
    fps = df.fps.sample(10000, random_state=42)
    # smiles = df.query('uniprot_id == "P00918"').smiles

    # smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 2000)
    # mols = smiles.apply(MolFromSmiles)
    # fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    # fps = pd.Series(fpgen.GetFingerprints(mols.values))

    measure_runtime(tanimoto_similarity_matrix_square, fps)

    res = tanimoto_similarity_matrix_square(fps[:1000])
    rd_res = jaccard_matrix_symm_rdkit(fps[:1000])
    assert np.allclose(rd_res, res)

    measure_runtime(tanimoto_similarity_matrix, fps, fps)

    res = tanimoto_similarity_matrix(fps[:1000], fps[-1000:])
    rd_res = jaccard_matrix_asymm_rdkit(fps[:1000], fps[-1000:])
    assert np.allclose(res, rd_res)


    arr = np.random.rand(200000).astype(np.float32)
    ref_arr = np.random.rand(5000).astype(np.float32)

    # Measure runtime for cross RMSD
    res1 = measure_runtime(calc_cross_rmsd, arr, ref_arr)
    res2 = calc_rmsd_np(arr, ref_arr)
    assert np.allclose(res1, res2)

    # arr = np.random.rand(200000)
    # ref_arr = np.random.rand(1000)
    #
    # # Measure runtime for cross RMSD
    # measure_runtime(calc_cross_rmsd, arr, ref_arr)
    #
    # # Measure runtime for numpy RMSD
    # measure_runtime(calc_rmsd_np, arr, ref_arr)
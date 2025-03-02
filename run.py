from tqdm import trange
import numpy as np
from time import time, sleep
import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from rdkit.DataStructs.cDataStructs import ExplicitBitVect, BulkTanimotoSimilarity
from chemivec import (tanimoto_similarity_matrix_square,
                      tanimoto_similarity_matrix,
                      calc_cross_rmsd,
                      calc_cross_diff_float32, calc_cross_diff, tanimoto_max_sim,
                      )
# from bitset.bitset import Bitset

def measure_runtime(func, *args, kwargs=None, n_runs=7, desc=None):
    """Measure the runtime of a function over multiple runs."""
    run_times = []
    for _ in trange(n_runs, desc=desc):
        if kwargs:
            tic = time()
            result = func(*args, **kwargs)
            toc = time()
        else:
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

def calc_cross_diff_np(arr, ref_arr):
    return np.abs(arr[:, np.newaxis] - ref_arr[np.newaxis, :])

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

def tanimoto_max_sim_np(arr1, arr2):
    return np.max(tanimoto_similarity_matrix(arr1, arr2), axis=1)

if __name__ == '__main__':

    """ ==== TANIMOTO SIMILARITY  ===="""

    df = pd.read_pickle('/home/ergot/projects/tadam-pipeline/data/processed/chembl_activities.pkl.zst')
    fps = df.fps.sample(10_000, random_state=42)
    # smiles = df.query('uniprot_id == "P00918"').smiles

    # smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 2000)
    # mols = smiles.apply(MolFromSmiles)
    # fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    # fps = pd.Series(fpgen.GetFingerprints(mols.values))

    # measure_runtime(tanimoto_similarity_matrix_square, fps, desc='tanimoto_square')
    #
    # res = tanimoto_similarity_matrix_square(fps[:1000])
    # rd_res = jaccard_matrix_symm_rdkit(fps[:1000])
    # assert np.allclose(rd_res, res)
    #
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, desc='tanimoto')
    #
    # res = tanimoto_similarity_matrix(fps[:1000], fps[-1000:])
    # rd_res = jaccard_matrix_asymm_rdkit(fps[:1000], fps[-1000:])
    # assert np.allclose(res, rd_res)
    #
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': 8}, desc='tanimoto 8')
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': 16}, desc='tanimoto 16')
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': -1}, desc='tanimoto -1')

    """ ==== TANIMOTO MAX SIM ===="""
    res = np.max(tanimoto_similarity_matrix(fps[:1000], fps[-1000:]), axis=1)
    res1 = tanimoto_max_sim(fps[:1000], fps[-1000:])
    assert np.allclose(res, res1)

    measure_runtime(tanimoto_max_sim, fps[:2000], fps[-2000:], kwargs={'n_jobs': -1}, n_runs=1, desc='tanimoto max sim')
    measure_runtime(tanimoto_max_sim_np, fps[:2000], fps[-2000:], n_runs=1, desc='tanimoto max sim np')






    # arr = np.random.rand(200_000).astype(np.float32)
    # ref_arr = np.random.rand(5000).astype(np.float32)
    #
    # # Measure runtime for cross RMSD
    # res1 = measure_runtime(calc_cross_rmsd, arr, ref_arr, desc='cross rmsd')
    # res2 = calc_rmsd_np(arr, ref_arr)
    # assert np.allclose(res1, res2)


    """ ==== CROSS DIFF ===="""

    # arr = np.random.rand(10_000).astype(np.float32)
    # ref_arr = np.random.rand(1000).astype(np.float32)
    # res1 = calc_cross_diff(arr, ref_arr)
    # res2 = calc_cross_diff_np(arr, ref_arr)
    # assert np.allclose(res1, res2)
    #
    # arr = np.random.rand(10_000).astype(np.float64)
    # ref_arr = np.random.rand(1000).astype(np.float64)
    # res1 = calc_cross_diff(arr, ref_arr)
    # res2 = calc_cross_diff_np(arr, ref_arr)
    # assert np.allclose(res1, res2)
    #
    # arr = np.random.randint(1000, size=10_000).astype(np.int32)
    # ref_arr = np.random.randint(1000, size=1000).astype(np.int32)
    # res1 = calc_cross_diff(arr, ref_arr)
    # res2 = calc_cross_diff_np(arr, ref_arr)
    # assert np.all(res1 == res2)



    # arr = np.random.rand(200_000).astype(np.float32)
    # ref_arr = np.random.rand(2000).astype(np.float32)
    # res2 = calc_cross_diff_np(arr, ref_arr)
    #
    # res1 = measure_runtime(calc_cross_diff, arr, ref_arr, kwargs={'n_jobs': 4}, n_runs=20, desc='cross diff 4')
    # assert np.allclose(res1, res2)
    # res1 = measure_runtime(calc_cross_diff, arr, ref_arr, kwargs={'n_jobs': 8}, n_runs=20, desc='cross diff 8')
    # assert np.allclose(res1, res2)
    # res1 = measure_runtime(calc_cross_diff, arr, ref_arr, kwargs={'n_jobs': 16}, n_runs=20, desc='cross diff 16')
    # assert np.allclose(res1, res2)
    # res1 = measure_runtime(calc_cross_diff, arr, ref_arr, kwargs={'n_jobs': -1}, n_runs=20, desc='cross diff -1')
    #
    # measure_runtime(calc_cross_diff_np, arr, ref_arr, desc='cross diff numpy')

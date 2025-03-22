from tqdm import trange
import numpy as np
from time import time, sleep
import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from rdkit.DataStructs.cDataStructs import ExplicitBitVect, BulkTanimotoSimilarity
from fpsim import (tanimoto_similarity_matrix_square,
                      tanimoto_similarity_matrix,
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

    """ ==== TANIMOTO SIMILARITY  ===="""

    # df = pd.read_pickle('/home/ergot/projects/tadam-pipeline/data/processed/chembl_activities.pkl.zst')
    # fps = df.fps.sample(10_000, random_state=42)
    # smiles = df.query('uniprot_id == "P00918"').smiles

    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 100000)
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))

    # measure_runtime(tanimoto_similarity_matrix_square, fps, desc='tanimoto_square')

    # res = tanimoto_similarity_matrix_square(fps)
    # rd_res = jaccard_matrix_symm_rdkit(fps)
    # assert np.allclose(rd_res, res)

    measure_runtime(tanimoto_similarity_matrix, fps[:1000], fps, desc='tanimoto')

    # measure_runtime(
    #     tanimoto_similarity_matrix, fps, fps,
    #     kwargs={'n_jobs': 1}, desc='tanimoto 1'
    # )

    # res = tanimoto_similarity_matrix(fps[:1000], fps[-1000:])
    # rd_res = jaccard_matrix_asymm_rdkit(fps[:1000], fps[-1000:])
    # assert np.allclose(res, rd_res)
    #
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': 8}, desc='tanimoto 8')
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': 16}, desc='tanimoto 16')
    # measure_runtime(tanimoto_similarity_matrix, fps, fps, kwargs={'n_jobs': -1}, desc='tanimoto -1')
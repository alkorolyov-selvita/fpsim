# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from typing import Dict, Any

cimport cython
from cython.operator cimport dereference as deref
from cython.parallel import prange, threadid, parallel
cimport numpy as cnp
cimport openmp
cnp.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, abort, calloc, abs
from libc.math cimport sqrt, fabs
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libcpp cimport bool as bool_t

import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit.DataStructs.cDataStructs import ExplicitBitVect as PyExplicitBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator


def calc_cross_diff_np(arr, ref_arr):
    return np.abs(arr[:, np.newaxis] - ref_arr[np.newaxis, :])


cdef void print_bitvec(ExplicitBitVect v):
    cdef unsigned int i
    for i in range(v.getNumBits()):
        printf("%d", v.getBit(i))
    printf("\n")


cdef void print_bitvec_ptr(ExplicitBitVect* v):
    cdef unsigned int i
    for i in range(v.getNumBits()):
        printf("%d", v.getBit(i))
    printf("\n")

cdef double jaccard_sim(ExplicitBitVect * v1, ExplicitBitVect * v2):
    cdef ExplicitBitVect intersection = deref(v1) & deref(v2)  # AND operation
    cdef unsigned int intersection_count = intersection.getNumOnBits()
    cdef unsigned int union_count = (deref(v1).getNumOnBits()
                                     + deref(v2).getNumOnBits()
                                     - intersection_count)

    # cdef ExplicitBitVect union = deref(v1) | deref(v2)  # OR operation
    # cdef unsigned int union_count = union.getNumOnBits()

    if union_count == 0:  # Avoid division by zero
        return 0.0

    return intersection_count / union_count


cdef float jaccard_sim_float(ExplicitBitVect * v1, ExplicitBitVect * v2) nogil:
    cdef unsigned int u_count, i_count
    i_count = (v1[0] & v2[0]).getNumOnBits()
    u_count = v1[0].getNumOnBits() + v2[0].getNumOnBits() - i_count

    if u_count == 0:
        return 0.0

    return i_count / u_count


# cdef void compute_pairwise_jaccard_similarities(cnp.ndarray[cnp.float32_t, ndim=2] jaccard_matrix, ExplicitBitVect** c_objects, int n):
#     for i in range(n - 1):
#         for j in range(i + 1, n):  # Only compute the upper triangle
#             jaccard_matrix[i, j] = jaccard_sim_float(c_objects[i], c_objects[j])
#             jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Symmetric matrix


def tanimoto_similarity_matrix_square(
        fps_array: pd.Series | np.ndarray,
        n_jobs: int = -1,
) -> np.ndarray[2, np.float32]:
    """
    Computes the Jaccard similarity matrix for a pandas.Series of ExplicitBitVect objects.

    Parameters:
        fps_array (pd.Series): A Series of ExplicitBitVect objects.

    Returns:
        np.ndarray: A 2D NumPy array containing pairwise Jaccard similarities.
    """
    if isinstance(fps_array, pd.Series):
        fps_array = fps_array.values

    cdef int n = fps_array.shape[0]
    cdef int i, j

    cdef ExplicitBitVect** c_objects = <ExplicitBitVect**>malloc(n * sizeof(ExplicitBitVect*))
    # cdef ExplicitBitVect** c_objects = <ExplicitBitVect**>PyMem_Malloc(n * sizeof(ExplicitBitVect*))

    # Extract C pointers from the Python objects
    cdef cnp.ndarray arr_np = fps_array
    for i in range(n):
        py_obj = <PyObject*>arr_np[i]
        c_objects[i] = extract_bitvec_ptr(py_obj)()

    # Allocate a NumPy array for the result
    cdef cnp.ndarray[cnp.float32_t, ndim=2] jaccard_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(jaccard_matrix, 1.0)

    # compute_pairwise_jaccard_similarities(jaccard_matrix, c_objects, n)

    # Compute pairwise Jaccard similarities
    for i in prange(n - 1, nogil=True, schedule='dynamic'):
        for j in range(i + 1, n):  # Only compute the upper triangle
            jaccard_matrix[i, j] = jaccard_sim_float(c_objects[i], c_objects[j])
            jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Symmetric matrix

    # Free the allocated C pointers
    free(c_objects)
    # PyMem_Free(c_objects)

    return jaccard_matrix


def tanimoto_similarity_matrix(
        fps_array1: pd.Series | np.ndarray,
        fps_array2: pd.Series | np.ndarray = None,
        n_jobs: int = -1,
) -> np.ndarray[2, np.float32]:
    """
    Computes the Jaccard similarity matrix for two pandas.Series of ExplicitBitVect objects.

    Parameters:
        fps_array1 (pd.Series or np.ndarray): A Series or array of ExplicitBitVect objects.
        fps_array2 (pd.Series or np.ndarray): A second Series or array of ExplicitBitVect objects.
        n_jobs (int): The number of parallel jobs to run. Default is -1, which means using all available CPUs.

    Returns:
        np.ndarray: A 2D NumPy array with shape (n, m), containing pairwise Jaccard similarities.
    """
    if fps_array2 is None:
        return tanimoto_similarity_matrix_square(fps_array1, n_jobs)

    # Ensure inputs are numpy arrays if they are pandas Series
    if isinstance(fps_array1, pd.Series):
        fps_array1 = fps_array1.values
    if isinstance(fps_array2, pd.Series):
        fps_array2 = fps_array2.values

    v1, v2 = fps_array1[0], fps_array2[0]
    if not isinstance(v1, PyExplicitBitVect) or not isinstance(v2, PyExplicitBitVect):
        raise TypeError(f'ExplicitBitVect expected, got {type(v1)} {type(v2)}')

    cdef int n = fps_array1.shape[0]
    cdef int m = fps_array2.shape[0]
    cdef int i, j
    cdef int n_jobs_c = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    # Allocate memory for C pointers for both arrays
    cdef ExplicitBitVect** c_objects1 = <ExplicitBitVect**>malloc(n * sizeof(ExplicitBitVect*))
    cdef ExplicitBitVect** c_objects2 = <ExplicitBitVect**>malloc(m * sizeof(ExplicitBitVect*))

    # Extract C pointers from the Python objects for the first array
    cdef cnp.ndarray arr_np1 = fps_array1
    for i in range(n):
        py_obj1 = <PyObject*>arr_np1[i]
        c_objects1[i] = extract_bitvec_ptr(py_obj1)()

    # Extract C pointers from the Python objects for the second array
    cdef cnp.ndarray arr_np2 = fps_array2
    for i in range(m):
        py_obj2 = <PyObject*>arr_np2[i]
        c_objects2[i] = extract_bitvec_ptr(py_obj2)()

    # Allocate a NumPy array for the result (asymmetric matrix)
    res = np.zeros((n, m), dtype=np.float32)
    cdef float[:, :] res_view = res

    # cdef cnp.ndarray[cnp.float32_t, ndim=2] jaccard_matrix = np.zeros((n, m), dtype=np.float32)

    # Compute pairwise Jaccard similarities (asymmetric matrix)
    for i in prange(n, nogil=True, schedule='dynamic', num_threads=n_jobs_c):
        for j in range(m):  # No symmetry, compute all pairs
            res_view[i, j] = jaccard_sim_float(c_objects1[i], c_objects2[j])

    # Free the allocated C pointers
    free(c_objects1)
    free(c_objects2)

    return res


cdef float max_arr(float[:] a) nogil:
    cdef int i
    cdef float max_val = a[0]
    cdef int n = a.shape[0]
    for i in range(1, n):
        if a[i] > max_val:
            max_val = a[i]
    return max_val

def tanimoto_max_sim(
        arr1: pd.Series | np.ndarray,
        arr2: pd.Series | np.ndarray,
        n_jobs: int = -1,
) -> np.ndarray[1, np.float32]:
    arr1 = arr1.values if isinstance(arr1, pd.Series) else arr1
    arr2 = arr2.values if isinstance(arr2, pd.Series) else arr2

    # check input type
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise TypeError(f'Expected numpy arrays, got {type(arr1)} {type(arr2)}')

    # check array dims
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError('Expected 1D arrays')

    # check array content
    v1, v2 = arr1[0], arr2[0]
    if not isinstance(v1, PyExplicitBitVect) or not isinstance(v2, PyExplicitBitVect):
        raise TypeError(f'ExplicitBitVect expected, got {type(v1)} {type(v2)}')

    n_jobs = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    cdef int i, j, batch_idx
    cdef int n = arr1.shape[0]
    cdef int m = arr2.shape[0]
    cdef int n_jobs_c = n_jobs

    # Allocate memory for C pointers for both arrays
    cdef ExplicitBitVect** c_bitvecs1 = <ExplicitBitVect**>malloc(n * sizeof(ExplicitBitVect*))
    cdef ExplicitBitVect** c_bitvecs2 = <ExplicitBitVect**>malloc(m * sizeof(ExplicitBitVect*))

    # Extract C pointers from the Python objects for the first array
    cdef cnp.ndarray arr_np1 = arr1
    for i in range(n):
        py_obj1 = <PyObject*>arr_np1[i]
        c_bitvecs1[i] = extract_bitvec_ptr(py_obj1)()

    # Extract C pointers from the Python objects for the second array
    cdef cnp.ndarray arr_np2 = arr2
    for i in range(m):
        py_obj2 = <PyObject*>arr_np2[i]
        c_bitvecs2[i] = extract_bitvec_ptr(py_obj2)()

    # Allocate a NumPy array for the result array of shape n
    res = np.empty(n, dtype=np.float32)
    cdef float[:] res_view = res
    tmp = np.empty((n_jobs, m), dtype=np.float32)
    cdef float[:, :] tmp_view = tmp
    cdef int tid


    with nogil, parallel(num_threads=n_jobs_c):
        # local_buf = <float *> malloc(sizeof(float) * n_jobs_c * m)
        # if local_buf is NULL:
        #     abort()

        # tmp_view = local_buf
        # Compute pairwise Jaccard similarities (asymmetric matrix)
        for i in prange(n, schedule='dynamic'):
            tid = threadid()
            for j in range(m):
                tmp_view[tid, j] = jaccard_sim_float(c_bitvecs1[i], c_bitvecs2[j])
                res_view[i] = max_arr(tmp_view[tid])

        # free(local_buf)
    return res


def calc_rmsd_float(arr: np.ndarray, ref_arr: np.ndarray):
    cdef int n = arr.shape[0]
    cdef int m = ref_arr.shape[0]

    # Allocate a NumPy array for the result (RMSD values)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] rmsd_values = np.zeros(n, dtype=np.float32)
    cdef float[:] rmsd_vals_view = rmsd_values

    cdef int i, j
    cdef float diff
    cdef float[:] arr_view = arr
    cdef float[:] ref_arr_view = ref_arr
    cdef float* local_buf

    # Compute the RMSD for each value in arr compared to ref_arr
    for i in prange(n, nogil=True):
    # for i in range(n):
        local_buf = <float*>calloc(1, sizeof(float))
        if local_buf is NULL:
            abort()
        for j in range(m):
            diff = arr_view[i] - ref_arr_view[j]
            local_buf[0] += diff ** 2
        rmsd_vals_view[i] = sqrt(local_buf[0] / m)
        free(local_buf)
    return rmsd_values

def calc_rmsd_double(arr, ref_arr):
    cdef int n = arr.shape[0]
    cdef int m = ref_arr.shape[0]

    # Allocate a NumPy array for the result (RMSD values)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rmsd_values = np.zeros(n, dtype=np.float64)
    cdef double[:] rmsd_vals_view = rmsd_values

    cdef int i, j
    cdef double diff
    cdef double[:] arr_view = arr
    cdef double[:] ref_arr_view = ref_arr
    cdef double* local_buf

    # Compute the RMSD for each value in arr compared to ref_arr
    for i in prange(n, nogil=True):
    # for i in range(n):
        local_buf = <double*>calloc(1, sizeof(double))
        if local_buf is NULL:
            abort()
        for j in range(m):
            diff = arr_view[i] - ref_arr_view[j]
            local_buf[0] += diff ** 2
        rmsd_vals_view[i] = sqrt(local_buf[0] / m)
        free(local_buf)
    return rmsd_values

def calc_cross_rmsd(arr: pd.Series | np.ndarray, ref_arr: pd.Series | np.ndarray) -> np.ndarray[1, float]:
    """
    Computes the RMSD (Root Mean Square Deviation) for each value in arr compared to ref_arr.

    Parameters:
        arr (pd.Series or np.ndarray): The first array of real numbers.
        ref_arr (pd.Series or np.ndarray): The second array of real numbers.

    Returns:
        np.ndarray: A 1D NumPy array containing the RMSD for each value in arr with respect to ref_arr.
    """
    # Ensure inputs are numpy arrays if they are pandas Series
    if isinstance(arr, pd.Series):
        arr = arr.values
    if isinstance(ref_arr, pd.Series):
        ref_arr = ref_arr.values
    if arr.dtype == np.float64 and ref_arr.dtype == np.float64:
        return calc_rmsd_double(arr, ref_arr)
    elif arr.dtype == np.float32 and ref_arr.dtype == np.float32:
        return  calc_rmsd_float(arr, ref_arr)
    else:
        raise TypeError(f'{arr.dtype} or {ref_arr.dtype} are not supported')

def calc_cross_diff_float32(arr: np.ndarray, ref_arr: np.ndarray, n_jobs=-1):
    cdef long n = arr.shape[0]
    cdef long m = ref_arr.shape[0]
    cdef long i, j
    cdef long n_jobs_cy = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    res = np.empty((n, m), dtype=np.float32)
    cdef float[:] arr_view = arr
    cdef float[:] ref_arr_view = ref_arr
    cdef float[:, :] res_view = res

    # for i in prange(n, nogil=True, schedule='dynamic', chunksize=2048, num_threads=n_jobs_cy):
    for i in prange(n, nogil=True, schedule='static', num_threads=n_jobs_cy):
    # for i in range(n):
        for j in range(m):
            res_view[i, j] = fabs(arr_view[i] - ref_arr_view[j])
    return res

def calc_cross_diff_float64(arr, ref_arr, n_jobs=-1):
    cdef long n = arr.shape[0]
    cdef long m = ref_arr.shape[0]
    cdef long i, j
    cdef long n_jobs_cy = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    res = np.empty((n, m), dtype=np.float64)
    cdef double[:] arr_view = arr
    cdef double[:] ref_arr_view = ref_arr
    cdef double[:, :] res_view = res

    # for i in prange(n, nogil=True, schedule='dynamic', chunksize=2048, num_threads=n_jobs_cy):
    for i in prange(n, nogil=True, schedule='static', num_threads=n_jobs_cy):
    # for i in range(n):
        for j in range(m):
            res_view[i, j] = fabs(arr_view[i] - ref_arr_view[j])

    return res

def calc_cross_diff_int32(arr, ref_arr, n_jobs=-1):
    cdef int n = arr.shape[0]
    cdef int m = ref_arr.shape[0]
    cdef int i, j
    cdef int n_jobs_cy = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    res = np.empty((n, m), dtype=np.int32)
    cdef int[:] arr_view = arr
    cdef int[:] ref_arr_view = ref_arr
    cdef int[:, :] res_view = res

    # for i in prange(n, nogil=True, schedule='dynamic', chunksize=2048, num_threads=n_jobs_cy):
    for i in prange(n, nogil=True, schedule='static', num_threads=n_jobs_cy):
    # for i in range(n):
        for j in range(m):
            res_view[i, j] = abs(arr_view[i] - ref_arr_view[j])
    return res

def calc_cross_diff_int64(arr, ref_arr, n_jobs=-1):
    cdef int n = arr.shape[0]
    cdef int m = ref_arr.shape[0]
    cdef int i, j
    cdef int n_jobs_cy = min(n_jobs, mp.cpu_count()) if n_jobs > 0 else mp.cpu_count()

    res = np.empty((n, m), dtype=np.int64)
    cdef long[:] arr_view = arr
    cdef long[:] ref_arr_view = ref_arr
    cdef long[:, :] res_view = res

    # for i in prange(n, nogil=True, schedule='dynamic', chunksize=2048, num_threads=n_jobs_cy):
    for i in prange(n, nogil=True, schedule='static', num_threads=n_jobs_cy):
    # for i in range(n):
        for j in range(m):
            res_view[i, j] = abs(arr_view[i] - ref_arr_view[j])
    return res



def calc_cross_diff(arr, ref_arr, n_jobs=-1):
    if isinstance(arr, pd.Series):
        arr = arr.values
    if isinstance(ref_arr, pd.Series):
        ref_arr = ref_arr.values

    if arr.dtype == np.float64 and ref_arr.dtype == np.float64:
        return calc_cross_diff_float64(arr, ref_arr, n_jobs=n_jobs)
    elif arr.dtype == np.float32 and ref_arr.dtype == np.float32:
        return calc_cross_diff_float32(arr, ref_arr, n_jobs=n_jobs)
    elif arr.dtype == np.int32 and ref_arr.dtype == np.int32:
        return calc_cross_diff_int32(arr, ref_arr, n_jobs=n_jobs)
    else:
        raise TypeError(f'{arr.dtype} or {ref_arr.dtype} are not supported')


def run_tests():
    cdef string s
    cdef bitset_t* bitset_cy = new bitset_t(4)
    bitset_cy.set(1)
    to_string(deref(bitset_cy), s)
    # printf("%s\n", s.c_str())

    cdef ExplicitBitVect u = ExplicitBitVect(4)
    cdef ExplicitBitVect v = ExplicitBitVect(4)
    u.setBit(1)
    v.setBit(1)
    v.setBit(3)

    print_bitvec(u)
    print_bitvec(v)
    cdef ExplicitBitVect res = u & v
    printf("res %d\n", res.getNumOnBits())
    printf("res %d\n", res.getNumBits())
    print_bitvec(res)

    # cdef unsigned int i
    # for i in range(u.getNumBits()):
    #     printf("%d", u.getBit(i))
    # printf("\n")

    cdef bitset_t* bs = v.dp_bits

    with nogil:
        printf("Bit set %d\n", bs.count())
        printf("Num blocks %d\n", bs.num_blocks())
        # printf("Block %d\n", bs.get_block(0))

    py_vec1 = PyExplicitBitVect(4)
    py_vec1.SetBit(1)
    py_vec1.SetBit(3)
    py_vec2 = PyExplicitBitVect(4)
    py_vec2.SetBit(1)
    py_vec2.SetBit(2)

    cdef ExplicitBitVect* bit_vec_ptr1 = extract_bitvec_ptr(<PyObject *> py_vec1)()
    cdef ExplicitBitVect* bit_vec_ptr2 = extract_bitvec_ptr(<PyObject *> py_vec2)()

    print('Extracted vectors:')
    print_bitvec_ptr(bit_vec_ptr1)
    print_bitvec_ptr(bit_vec_ptr2)

    print('Logic AND')
    print_bitvec(deref(bit_vec_ptr1) & deref(bit_vec_ptr2))
    print('Logic OR')
    print_bitvec(deref(bit_vec_ptr1) | deref(bit_vec_ptr2))

    printf('Jaccard sim %.3f\n', jaccard_sim(bit_vec_ptr1, bit_vec_ptr2))

    smiles = pd.Series(['CC', 'CO', 'CN', 'CF'])
    mols = smiles.apply(MolFromSmiles)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=16)
    fps = pd.Series(fpgen.GetFingerprints(mols.values), index=smiles)

    print('Fingerprints')
    print(fps)
    print(fps.shape)
    print(fps.dtype)
    print(type(fps.iat[0]))

    arr = tanimoto_similarity_matrix_square(fps)
    print(arr)

    arr = tanimoto_similarity_matrix(fps[:3], fps[2:])
    print(arr)

    res1 = tanimoto_max_sim(fps[:3], fps[2:])
    res2 = np.max(arr, axis=1)
    assert np.allclose(res1, res2)

    print('RMSD')
    a1 = np.random.rand(8).astype(np.float32)
    a2 = np.random.rand(12).astype(np.float32)
    print(calc_cross_rmsd(a1, a2).shape, a1.shape)
    print(np.round(calc_cross_rmsd(a1, a2), 2))




    a1 = np.array([1, 2], dtype=np.float32)
    a2 = np.array([1, 2, 3], dtype=np.float32)
    assert np.allclose(calc_cross_diff_float32(a1, a2), calc_cross_diff_np(a1, a2))

    a1 = np.array([1, 2], dtype=np.float64)
    a2 = np.array([1, 2, 3], dtype=np.float64)
    assert np.allclose(calc_cross_diff_float64(a1, a2), calc_cross_diff_np(a1, a2))

    a1 = np.array([1, 2], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    assert np.all(calc_cross_diff_int32(a1, a2) == calc_cross_diff_np(a1, a2))

    a1 = np.array([1, 2], dtype=np.int64)
    a2 = np.array([1, 2, 3], dtype=np.int64)
    assert np.all(calc_cross_diff_int64(a1, a2) == calc_cross_diff_np(a1, a2))


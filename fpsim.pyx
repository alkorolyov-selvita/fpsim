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

cdef double jaccard_sim_double(ExplicitBitVect * v1, ExplicitBitVect * v2):
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


cdef inline float jaccard_sim_float(ExplicitBitVect * v1, ExplicitBitVect * v2) nogil:
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
        # printf("%d\n",&c_objects1[i])

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

    printf('Jaccard sim %.3f\n', jaccard_sim_double(bit_vec_ptr1, bit_vec_ptr2))

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

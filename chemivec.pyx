# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


cimport cython
from cython.operator cimport dereference as deref
from cython.parallel import prange
cimport numpy as cnp
cnp.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, abort, calloc
from libc.math cimport sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libcpp cimport bool as bool_t

import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit.DataStructs.cDataStructs import ExplicitBitVect as PyExplicitBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator


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
    return i_count / u_count


# cdef void compute_pairwise_jaccard_similarities(cnp.ndarray[cnp.float32_t, ndim=2] jaccard_matrix, ExplicitBitVect** c_objects, int n):
#     for i in range(n - 1):
#         for j in range(i + 1, n):  # Only compute the upper triangle
#             jaccard_matrix[i, j] = jaccard_sim_float(c_objects[i], c_objects[j])
#             jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Symmetric matrix


def tanimoto_similarity_matrix_square(
        fps_array: pd.Series | np.ndarray
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
        fps_array2: pd.Series | np.ndarray = None
) -> np.ndarray[2, np.float32]:
    """
    Computes the Jaccard similarity matrix for two pandas.Series of ExplicitBitVect objects.

    Parameters:
        fps_array1 (pd.Series or np.ndarray): A Series or array of ExplicitBitVect objects.
        fps_array2 (pd.Series or np.ndarray): A second Series or array of ExplicitBitVect objects.

    Returns:
        np.ndarray: A 2D NumPy array with shape (n, m), containing pairwise Jaccard similarities.
    """
    if fps_array2 is None:
        return tanimoto_similarity_matrix_square(fps_array1)

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
    cdef cnp.ndarray[cnp.float32_t, ndim=2] jaccard_matrix = np.zeros((n, m), dtype=np.float32)

    # Compute pairwise Jaccard similarities (asymmetric matrix)
    for i in prange(n, nogil=True, schedule='dynamic'):
        for j in range(m):  # No symmetry, compute all pairs
            jaccard_matrix[i, j] = jaccard_sim_float(c_objects1[i], c_objects2[j])

    # Free the allocated C pointers
    free(c_objects1)
    free(c_objects2)

    return jaccard_matrix


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

    print(fps)
    print(fps.shape)
    print(fps.dtype)
    print(type(fps.iat[0]))

    arr = tanimoto_similarity_matrix_square(fps)
    print(arr)

    arr = tanimoto_similarity_matrix(fps[:3], fps[2:])
    print(arr)

    a1 = np.random.rand(8).astype(np.float32)
    a2 = np.random.rand(12).astype(np.float32)
    print(calc_cross_rmsd(a1, a2).shape, a1.shape)
    print(np.round(calc_cross_rmsd(a1, a2), 2))

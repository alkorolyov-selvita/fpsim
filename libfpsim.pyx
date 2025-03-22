# distutils: language = c++
# distutils: extra_compile_args = -std=c++20
# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import os

import numpy as np
import pandas as pd
import cupy as cp
cimport numpy as np
cimport cython
from cython.parallel import prange
from cpython.ref cimport PyObject
from numpy cimport import_array
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport exit, malloc, free
from libc.stdio cimport printf
from libcpp cimport bool

from rdkit.DataStructs.cDataStructs import ExplicitBitVect as PyExplicitBitVect

import_array()  # Required for NumPy C API

""" ================== DEFINITIONS ======================== """
# include CPU popcnt functions built in GCC for all posix systems
# using -march=native GCC flag will use best CPU instruction available
# Use __builtin_popcountll for unsigned 64-bit integers
# equivalent to https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-builtin.cpp#L23
cdef extern int __builtin_popcountll(unsigned long long) nogil


# Define the Boost dynamic_bitset interface first, before it's used
cdef extern from "<boost/dynamic_bitset.hpp>" namespace "boost":
    cdef cppclass dynamic_bitset[T]:
        size_t size()
        size_t num_blocks() const

# Although it is a template, we really only want to use it one way,
# instantiated with size_t, so we typedef the particular case we want.
ctypedef dynamic_bitset[uint64_t] bitset_t

# We need to forward declare the iterator_tag from standard library
cdef extern from "<iterator>" namespace "std":
    cdef cppclass output_iterator_tag:
        pass

# Instead of using the custom iterator directly, let's create a wrapper function in C++
cdef extern from *:
    """
    #include <iterator>
    #include <boost/dynamic_bitset.hpp>

    // Custom output iterator that writes directly to a memory location
    template <typename T>
    class numpy_output_iterator {
    private:
        T* data;
        size_t index;

    public:
        // Iterator type definitions required by C++ standard
        typedef std::output_iterator_tag iterator_category;
        typedef void value_type;
        typedef void difference_type;
        typedef void pointer;
        typedef void reference;

        // Constructor
        numpy_output_iterator(T* data_ptr) : data(data_ptr), index(0) {}

        // Dereference operator - returns a proxy that can be assigned to
        numpy_output_iterator& operator*() { return *this; }

        // Assignment operator for the proxy
        numpy_output_iterator& operator=(const T& value) {
            data[index] = value;
            return *this;
        }

        // Pre-increment
        numpy_output_iterator& operator++() {
            ++index;
            return *this;
        }

        // Post-increment
        numpy_output_iterator operator++(int) {
            numpy_output_iterator tmp = *this;
            ++index;
            return tmp;
        }

        // Equality operator
        bool operator==(const numpy_output_iterator& other) const {
            return data == other.data && index == other.index;
        }

        // Inequality operator
        bool operator!=(const numpy_output_iterator& other) const {
            return !(*this == other);
        }
    };

    // Wrapper function that hides the custom iterator
    // This function creates the iterator and uses it with to_block_range internally
    void to_block_range_direct(const boost::dynamic_bitset<size_t>& bs, uint64_t* data) {
        numpy_output_iterator<uint64_t> it(data);
        boost::to_block_range(bs, it);
    }
    """
    # Now we only need to declare the wrapper function using our typedef bitset_t
    void to_block_range_direct(const bitset_t& bs, uint64_t * data) nogil


# Define the ExplicitBitVect interface
cdef extern from "rdkit/DataStructs/ExplicitBitVect.h":
    cdef cppclass ExplicitBitVect:
        unsigned int getNumOnBits() nogil const
        ExplicitBitVect operator &(const ExplicitBitVect& other) nogil
        bitset_t *dp_bits

# Define the boost::python::extract interface
cdef extern from "<boost/python/extract.hpp>" namespace "boost::python":
    cdef cppclass extract[T]:
        extract(PyObject * obj)
        T operator()() const
        bool check() const

ctypedef extract[ExplicitBitVect *] extract_bitvec_ptr

""" ================== BITVEC TO PACKED NUMPY ======================== """


# Function to convert ExplicitBitVect to numpy array using our custom iterator
def bitvec_to_numpy(py_vec):
    """
    Convert RDKit ExplicitBitVect directly to numpy array of uint64
    using a custom output iterator for zero-copy conversion

    Parameters:
    -----------
    py_vec : RDKit ExplicitBitVect
        The bit vector to convert

    Returns:
    --------
    numpy.ndarray
        A numpy array of uint64 values representing the bit vector
    """
    # Extract the C++ pointer from the Python object
    cdef ExplicitBitVect * bit_vec_ptr = extract_bitvec_ptr(<PyObject *> py_vec)()

    # Get the dynamic_bitset from the ExplicitBitVect
    cdef bitset_t * bs = bit_vec_ptr.dp_bits

    # Determine the number of blocks (uint64 values) needed
    cdef size_t num_blocks = bs.num_blocks()

    # Create a numpy array to hold the result
    cdef np.ndarray[np.uint64_t, ndim=1] arr = np.zeros(num_blocks, dtype=np.uint64)

    # Get a pointer to the array's data
    cdef uint64_t * arr_data = <uint64_t *> arr.data

    # Use our wrapper function to fill the numpy array directly
    with nogil:
        to_block_range_direct(bs[0], arr_data)

    return arr

# Function to convert an array of ExplicitBitVects to a 2D numpy array
def bitvec_arr_to_numpy(py_vecs):
    """
    Convert an array of RDKit ExplicitBitVect objects to a 2D numpy array
    using a custom output iterator for efficient conversion.

    Parameters:
    -----------
    py_vecs : list or iterable
        List of RDKit ExplicitBitVect objects

    Returns:
    --------
    numpy.ndarray
        A 2D numpy array of uint64 values representing the bit vectors,
        with shape (n_vectors, num_blocks)
    """

    # Get the first vector to extract metadata
    cdef ExplicitBitVect * first_vec_ptr = extract_bitvec_ptr(<PyObject *> py_vecs[0])()
    cdef bitset_t * first_bs = first_vec_ptr.dp_bits

    # Get number of blocks directly from the bitset
    cdef size_t num_blocks = first_bs.num_blocks()

    # Create a 2D numpy array to hold all results
    cdef np.ndarray[np.uint64_t, ndim=2] result = np.zeros(
        (len(py_vecs), num_blocks), dtype=np.uint64
    )

    # Process each bit vector
    cdef int i
    cdef ExplicitBitVect * bit_vec_ptr
    cdef bitset_t * bs
    cdef uint64_t * row_data

    for i in range(len(py_vecs)):
        # Extract the C++ pointer from the Python object
        bit_vec_ptr = extract_bitvec_ptr(<PyObject *> py_vecs[i])()
        # Get the dynamic_bitset from the ExplicitBitVect
        bs = bit_vec_ptr.dp_bits

        # Get a pointer to the current row's data
        row_data = <uint64_t *> &result[i, 0]

        # Use our wrapper function to fill the numpy array directly
        with nogil:
            to_block_range_direct(bs[0], row_data)

    return result

""" ================== SIMILARITY ON PACKED NUMPY FINGERPRINTS ======================== """

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline float _tanimoto_sim(uint32_t int_count, uint32_t count_query, uint32_t count_other) nogil:
    cdef double t_coeff = 0.0
    t_coeff = count_query + count_other - int_count
    if t_coeff != 0.0:
        t_coeff = int_count / t_coeff
    return t_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline uint32_t _i_popcount(uint64_t[:] query, uint64_t[:] other) nogil:
    cdef uint32_t int_count = 0
    cdef uint32_t j
    for j in range(0, query.shape[0], 4):
        int_count += __builtin_popcountll(other[j] & query[j])
        int_count += __builtin_popcountll(other[j + 1] & query[j + 1])
        int_count += __builtin_popcountll(other[j + 2] & query[j + 2])
        int_count += __builtin_popcountll(other[j + 3] & query[j + 3])
    return int_count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline uint32_t _calculate_popcount(uint64_t[:] fp) nogil:
    cdef uint32_t count = 0
    cdef uint32_t j
    for j in range(fp.shape[0]):
        count += __builtin_popcountll(fp[j])
    return count



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[np.uint32_t, ndim=1] get_popcounts(uint64_t[:, :] fps):
    cdef uint32_t i
    cdef uint32_t n = fps.shape[0]
    cdef np.ndarray[np.uint32_t, ndim=1] res = np.zeros(n, dtype=np.uint32)

    with nogil:
        for i in range(n):
            res[i] = _calculate_popcount(fps[i])
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[np.float32_t, ndim=2] tanimoto_matrix_numpy(uint64_t[:, :] fps1, uint64_t[:, :] fps2, int n_jobs=1):
    """
    Calculate the Tanimoto similarity matrix between two sets of fingerprints.

    Parameters:
    -----------
    fps1 : 2D array of uint64
        First set of fingerprints
    fps2 : 2D array of uint64
        Second set of fingerprints

    Returns:
    --------
    np.ndarray : 2D matrix of float32
        Matrix of Tanimoto coefficients where result[i,j] is the 
        similarity between fps1[i] and fps2[j]
    """
    cdef int num_threads = n_jobs if n_jobs > 0 else os.cpu_count()
    cdef uint32_t n_fps1 = fps1.shape[0]
    cdef uint32_t n_fps2 = fps2.shape[0]

    # Create numpy arrays
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((n_fps1, n_fps2), dtype=np.float32)
    cdef np.ndarray[np.uint32_t, ndim=1] popcounts1 = np.zeros(n_fps1, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] popcounts2 = np.zeros(n_fps2, dtype=np.uint32)

    # Define views
    cdef float[:,:] res_view = result
    cdef uint32_t[:] popcnt_view1 = popcounts1
    cdef uint32_t[:] popcnt_view2 = popcounts2

    cdef uint32_t i, j
    cdef uint32_t int_count

    with nogil:
        # _tanimoto_sim_matrix(fps1, fps2, popcnt_view1, popcnt_view2, res_view)

        # Precompute all popcounts
        for i in range(n_fps1):
            popcounts1[i] = _calculate_popcount(fps1[i])

        for j in range(n_fps2):
            popcounts2[j] = _calculate_popcount(fps2[j])


        # Calculate similarity matrix
        for i in prange(n_fps1, num_threads=num_threads):
            # print_progress(i, n_fps1)
            for j in range(n_fps2):
                int_count = _i_popcount(fps1[i], fps2[j])
                result[i, j] = _tanimoto_sim(int_count, popcounts1[i], popcounts2[j])

    return result



""" ================== SIMILARITY BITVEC FINGERPINTS ========== """

cdef inline float _tanimoto_sim_bitvec(ExplicitBitVect * v1, ExplicitBitVect * v2) nogil:
    cdef unsigned int u_count, i_count
    i_count = (v1[0] & v2[0]).getNumOnBits()
    u_count = v1[0].getNumOnBits() + v2[0].getNumOnBits() - i_count

    if u_count == 0:
        return 0.0

    return i_count / u_count

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def tanimoto_matrix_bitvec(
        fps_array1: pd.Series | np.ndarray,
        fps_array2: pd.Series | np.ndarray,
        n_jobs: int = -1,
):
    """
    Computes the Jaccard similarity matrix for two pandas.Series of ExplicitBitVect objects.

    Parameters:
        fps_array1 (pd.Series or np.ndarray): A Series or array of ExplicitBitVect objects.
        fps_array2 (pd.Series or np.ndarray): A second Series or array of ExplicitBitVect objects.
        n_jobs (int): The number of parallel jobs to run. Default is -1, which means using all available CPUs.

    Returns:
        np.ndarray: A 2D NumPy array with shape (n, m), containing pairwise Jaccard similarities.
    """

    # Ensure inputs are numpy arrays if they are pandas Series
    if isinstance(fps_array1, pd.Series):
        fps_array1 = fps_array1.values
    if isinstance(fps_array2, pd.Series):
        fps_array2 = fps_array2.values

    v1, v2 = fps_array1[0], fps_array2[0]
    if not isinstance(v1, PyExplicitBitVect) or not isinstance(v2, PyExplicitBitVect):
        raise TypeError(f'ExplicitBitVect expected, got {type(v1)} {type(v2)}')

    cdef int n_fps1 = fps_array1.shape[0]
    cdef int n_fps2 = fps_array2.shape[0]
    cdef int i, j
    cdef int num_threads = min(n_jobs, os.cpu_count()) if n_jobs > 0 else os.cpu_count()

    # Allocate memory for C pointers for both arrays
    cdef ExplicitBitVect** bit_vecs1 = <ExplicitBitVect**>malloc(n_fps1 * sizeof(ExplicitBitVect*))
    cdef ExplicitBitVect** bit_vecs2 = <ExplicitBitVect**>malloc(n_fps2 * sizeof(ExplicitBitVect*))

    # Extract C pointers from the Python objects for the first array
    cdef np.ndarray arr_np1 = fps_array1
    for i in range(n_fps1):
        py_obj1 = <PyObject*>arr_np1[i]
        bit_vecs1[i] = extract_bitvec_ptr(py_obj1)()

    # Extract C pointers from the Python objects for the second array
    cdef np.ndarray arr_np2 = fps_array2
    for i in range(n_fps2):
        py_obj2 = <PyObject*>arr_np2[i]
        bit_vecs2[i] = extract_bitvec_ptr(py_obj2)()

    # Allocate a NumPy array for the result (asymmetric matrix)
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros((n_fps1, n_fps2), dtype=np.float32)

    # Compute pairwise Jaccard similarities (asymmetric matrix)
    for i in prange(n_fps1, nogil=True, schedule='dynamic', num_threads=num_threads):
        for j in range(n_fps2):  # No symmetry, compute all pairs
            res[i, j] = _tanimoto_sim_bitvec(bit_vecs1[i], bit_vecs2[j])

    # Free the allocated C pointers
    free(bit_vecs1)
    free(bit_vecs2)

    return res

""" ========= SIMILARITY GPU ON NUMPY FINGERPRINTS =========== """
def similarity_matrix_cpu(fps1, fps2, popcnts1, popcnts2):
    """Calculate Tanimoto similarity matrix on CPU for verification."""
    n_fps1, fp_len = fps1.shape
    n_fps2 = fps2.shape[0]
    result = np.zeros((n_fps1, n_fps2), dtype=np.float32)

    for i in range(n_fps1):
        for j in range(n_fps2):
            # Calculate intersection (common bits)
            common_bits = 0
            for k in range(fp_len):
                common_bits += bin(fps1[i, k] & fps2[j, k]).count('1')

            # Calculate Tanimoto coefficient
            union_bits = popcnts1[i] + popcnts2[j] - common_bits
            if union_bits > 0:
                result[i, j] = common_bits / union_bits

    return result

def _similarity_matrix_gpu(fps1, fps2, popcnts1, popcnts2):
    """Calculate Tanimoto similarity matrix between two sets of fingerprints using GPU.

    Parameters
    ----------
    fps1 : numpy.ndarray
        First set of fingerprints packed as uint64 arrays with shape (n, fp_len)
    fps2 : numpy.ndarray
        Second set of fingerprints packed as uint64 arrays with shape (m, fp_len)
    popcnts1 : numpy.ndarray
        Popcount values for fps1 with shape (n,)
    popcnts2 : numpy.ndarray
        Popcount values for fps2 with shape (m,)

    Returns
    -------
    numpy.ndarray
        Similarity matrix with shape (n, m)
    """
    # Get dimensions
    n_fps1 = len(fps1)
    n_fps2 = len(fps2)
    fp_len = fps1.shape[1]  # Number of uint64 elements per fingerprint

    # Transfer data to GPU
    cuda_fps1 = cp.asarray(fps1, dtype=cp.uint64)
    cuda_fps2 = cp.asarray(fps2, dtype=cp.uint64)
    cuda_popcnts1 = cp.asarray(popcnts1, dtype=cp.uint64)
    cuda_popcnts2 = cp.asarray(popcnts2, dtype=cp.uint64)

    # Create output matrix
    similarity_matrix = cp.zeros((n_fps1, n_fps2), dtype=cp.float32)

    # Define the raw CUDA kernel for matrix calculation
    raw_kernel = r"""
    extern "C" __global__
    void taniMatrix(const unsigned long long int* fps1,
                   const unsigned long long int* popcnts1,
                   const unsigned long long int* fps2,
                   const unsigned long long int* popcnts2,
                   float* similarity_matrix,
                   const int fp_len,
                   const int n_fps2) {

        // Get the index of fps1 (row index)
        int idx1 = blockIdx.x;
        // Get the index of fps2 (column index)
        int idx2 = threadIdx.x + blockIdx.y * blockDim.x;

        // Check if we're within bounds
        if (idx2 >= n_fps2) return;

        // Calculate base addresses
        const unsigned long long int* fp1 = fps1 + idx1 * fp_len;
        const unsigned long long int* fp2 = fps2 + idx2 * fp_len;

        // Calculate common bits (popcount of bitwise AND)
        int common_bits = 0;
        for (int i = 0; i < fp_len; i++) {
            common_bits += __popcll(fp1[i] & fp2[i]);
        }

        // Calculate Tanimoto coefficient
        float coeff = 0.0f;
        float union_bits = popcnts1[idx1] + popcnts2[idx2] - common_bits;

        if (union_bits != 0.0f) {
            coeff = common_bits / union_bits;
        }

        // Store result in the matrix
        similarity_matrix[idx1 * n_fps2 + idx2] = coeff;
    }
    """

    # Compile the kernel
    tani_matrix_kernel = cp.RawKernel(
        raw_kernel,
        name="taniMatrix",
        options=("-std=c++14",),
    )

    # Configure grid and block dimensions
    # Each block processes one fingerprint from fps1 against multiple fps2
    threads_per_block = 256  # Adjust based on GPU capabilities
    blocks_y = (n_fps2 + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    tani_matrix_kernel(
        (n_fps1, blocks_y),  # Grid dimensions
        (threads_per_block,),  # Block dimensions
        (
            cuda_fps1,
            cuda_popcnts1,
            cuda_fps2,
            cuda_popcnts2,
            similarity_matrix,
            fp_len,
            n_fps2,
        ),
    )

    # Transfer results back to CPU
    return cp.asnumpy(similarity_matrix)


def similarity_matrix_gpu(fps1, fps2, popcnts1, popcnts2, batch_size=1024):
    """Calculate Tanimoto similarity matrix with batching for large datasets.

    Parameters
    ----------
    fps1 : numpy.ndarray
        First set of fingerprints packed as uint64 arrays
    fps2 : numpy.ndarray
        Second set of fingerprints packed as uint64 arrays
    popcnts1 : numpy.ndarray
        Popcount values for fps1
    popcnts2 : numpy.ndarray
        Popcount values for fps2
    batch_size : int
        Number of fingerprints to process in each batch

    Returns
    -------
    numpy.ndarray
        Similarity matrix
    """
    n_fps1 = len(fps1)
    n_fps2 = len(fps2)

    # Create output matrix
    result_matrix = np.zeros((n_fps1, n_fps2), dtype=np.float32)

    # Process in batches
    for i in range(0, n_fps1, batch_size):
        batch_end_i = min(i + batch_size, n_fps1)
        fps1_batch = fps1[i:batch_end_i]
        popcnts1_batch = popcnts1[i:batch_end_i]

        for j in range(0, n_fps2, batch_size):
            batch_end_j = min(j + batch_size, n_fps2)
            fps2_batch = fps2[j:batch_end_j]
            popcnts2_batch = popcnts2[j:batch_end_j]

            # Calculate similarity matrix for this batch
            batch_matrix = _similarity_matrix_gpu(
                fps1_batch, fps2_batch, popcnts1_batch, popcnts2_batch
            )

            # Store results in the final matrix
            result_matrix[i:batch_end_i, j:batch_end_j] = batch_matrix

    return result_matrix




""" ================== MISC FUNCTIONS ======================== """


def check_unsigned_long_size():
    cdef unsigned long ul
    cdef size_t size = sizeof(ul) * 8

    print("Size of unsigned long: %d bits\n" % size)

    # Check if 64-bit (unsigned long should be 8 bytes)
    if size != 64:
        print("ERROR: This code requires a 64-bit system (unsigned long = 8 bytes).\n")
        print("Current system has unsigned long = %d bits.\n" % size)
        # Abort the program
        exit(1)

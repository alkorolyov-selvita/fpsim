# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


import numpy as np
cimport numpy as np
cimport cython
from cpython.ref cimport PyObject
from numpy cimport npy_intp, import_array
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport exit
from libc.stdio cimport printf
from libcpp cimport bool

import_array()  # Required for NumPy C API

""" ================== DEFINITIONS ======================== """

# include CPU popcnt functions built in GCC for all posix systems
# using -march=native GCC flag will use best CPU instruction available
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
        bitset_t *dp_bits

# Define the boost::python::extract interface
cdef extern from "<boost/python/extract.hpp>" namespace "boost::python":
    cdef cppclass extract[T]:
        extract(PyObject * obj)
        T operator()() const
        bool check() const

ctypedef extract[ExplicitBitVect *] extract_bitvec_ptr

""" ================== MAIN FUNCTIONS ======================== """

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline float _tanimoto_coeff(uint32_t int_count, uint32_t count_query, uint32_t count_other) nogil:
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
        # Use __builtin_popcountll for unsigned 64-bit integers (fps j+ 1 in other to skip the mol_id)
        # equivalent to https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-builtin.cpp#L23
        int_count += __builtin_popcountll(other[j + 1] & query[j])
        int_count += __builtin_popcountll(other[j + 2] & query[j + 1])
        int_count += __builtin_popcountll(other[j + 3] & query[j + 2])
        int_count += __builtin_popcountll(other[j + 4] & query[j + 3])
    return int_count


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
    if not py_vecs:
        return np.zeros((0, 0), dtype=np.uint64)

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

""" ================== TEST FUNCTIONS ======================== """

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


def timeit(func, *args, kwargs=None, n_runs=7, desc=None):
    """Measure the runtime of a function over multiple runs."""
    from tqdm import trange
    from time import time

    run_times = []
    if kwargs is None:
        kwargs = {}
    for _ in trange(n_runs, desc=desc):
        tic = time()
        result = func(*args, **kwargs)
        toc = time()
        run_times.append(toc - tic)
    run_times = np.array(run_times)
    print(f'{n_runs} runs {run_times.mean():.3f} ± {run_times.std():.3f} s')
    return result


# Test functions for BitVect to numpy conversions
def test_bitvec_to_numpy():
    """
    Test the direct conversion of a single ExplicitBitVect to a numpy array
    """
    from rdkit import DataStructs

    # Create an ExplicitBitVect
    py_vec = DataStructs.ExplicitBitVect(128)
    py_vec.SetBit(1)
    py_vec.SetBit(65)

    # Convert to numpy array using our direct method
    numpy_arr = bitvec_to_numpy(py_vec)
    print('Numpy array:', numpy_arr)
    print('Data type:', numpy_arr.dtype)

    # Verify bits are set correctly
    # Bit 1 should be in the first uint64 block (block 0)
    # Bit 65 should be in the second uint64 block (block 1)
    assert numpy_arr[0] == 2  # 2^1 = 2
    assert numpy_arr[1] == 2  # 2^(65-64) = 2^1 = 2

    return numpy_arr

# Test for the array conversion function
def test_bitvec_arr_to_numpy():
    """
    Test the conversion of multiple ExplicitBitVect objects to a 2D numpy array
    """
    from rdkit import DataStructs

    # Create multiple ExplicitBitVects
    num_vecs = 5
    py_vecs = []
    for i in range(num_vecs):
        bv = DataStructs.ExplicitBitVect(128)
        # Set different bits for each vector
        bv.SetBit(i)  # Sets bit i
        bv.SetBit(64 + i)  # Sets bit 64+i
        py_vecs.append(bv)

    # Convert to 2D numpy array
    numpy_arr_2d = bitvec_arr_to_numpy(py_vecs)
    print('2D Numpy array shape:', numpy_arr_2d.shape)
    print('2D Numpy array:')
    print(numpy_arr_2d)

    # Verify bits are set correctly
    for i in range(num_vecs):
        # Verify first block (bits 0-63)
        assert numpy_arr_2d[i, 0] == (1 << i)
        # Verify second block (bits 64-127)
        assert numpy_arr_2d[i, 1] == (1 << i)

    return numpy_arr_2d



# Simple benchmark function
def benchmark_bitvec_arr_to_numpy():
    import random
    from tqdm import trange
    from rdkit.Chem import DataStructs

    # Function to create random bit vectors
    def create_random_bitvecs(n, size):
        """Create n random bit vectors of given size"""
        vectors = []
        for _ in trange(n):
            bv = DataStructs.ExplicitBitVect(size)
            # Set ~10% of bits randomly
            # on_bits = random.sample(range(size), size // 10)
            # for bit in on_bits:
            #     bv.SetBit(bit)
            vectors.append(bv)
        return vectors

    # Parameters
    n_vectors = 100000
    vector_size = 2048

    vectors = create_random_bitvecs(n_vectors, vector_size)

    timeit(bitvec_arr_to_numpy, vectors)



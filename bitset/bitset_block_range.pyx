
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from cpython.ref cimport PyObject
import numpy as np
cimport numpy as np
from numpy cimport npy_intp, import_array
from libc.stdint cimport uint64_t

import_array()  # Required for NumPy C API

from libcpp cimport bool

# Define the Boost dynamic_bitset interface first, before it's used
cdef extern from "<boost/dynamic_bitset.hpp>" namespace "boost":
    cdef cppclass dynamic_bitset[T]:
        size_t num_blocks() const

# Although it is a template, we really only want to use it one way,
# instantiated with size_t, so we typedef the particular case we want.
ctypedef dynamic_bitset[size_t] bitset_t

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

# Function to convert ExplicitBitVect to numpy array using our custom iterator
def bitvec_to_numpy_direct(py_vec):
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



# Example usage
def example():
    from rdkit import DataStructs

    # Create an ExplicitBitVect
    py_vec = DataStructs.ExplicitBitVect(128)
    py_vec.SetBit(1)
    py_vec.SetBit(65)

    # Convert to numpy array using our direct method
    numpy_arr = bitvec_to_numpy_direct(py_vec)
    print('Numpy array:', numpy_arr)
    print('Data type:', numpy_arr.dtype)

    # Verify bits are set correctly
    # Bit 1 should be in the first uint64 block (block 0)
    # Bit 65 should be in the second uint64 block (block 1)
    assert numpy_arr[0] == 2  # 2^1 = 2
    assert numpy_arr[1] == 2  # 2^(65-64) = 2^1 = 2

    return numpy_arr
from cpython.ref cimport PyObject
from libcpp.string cimport string


cdef extern from "<boost/dynamic_bitset.hpp>" namespace "boost":
    # We're wrapping a Template class from boost.
    cdef cppclass dynamic_bitset[T]:
        dynamic_bitset(size_t) nogil
        void resize(size_t)
        void set(size_t)
        void reset(size_t)
        void flip(size_t)

        size_t size()
        bint test(size_t)
        bint empty()
        bint all()
        bint any()
        bint none()
        size_t count() nogil
        bint is_subset_of(dynamic_bitset[T]&)
        bint is_proper_subset_of(dynamic_bitset[T]&)
        bint intersects(dynamic_bitset[T]& a)

    # This function is templated, so will automatically work.
    cdef void to_string(dynamic_bitset[size_t], string s)

# Although it is a template, we really only want to use it one way,
# instantiated with size_t, so we typedef the particular case we want.
ctypedef dynamic_bitset[size_t] bitset_t


cdef extern from "rdkit/DataStructs/ExplicitBitVect.h":
    cdef cppclass ExplicitBitVect:
        ExplicitBitVect() nogil
        ExplicitBitVect(unsigned int size) nogil

        unsigned int getNumBits() nogil const
        unsigned int getNumOnBits() nogil const
        void setBit(unsigned int bitId)
        bint getBit(unsigned int bitId) const
        string toString() const

        ExplicitBitVect operator^(const ExplicitBitVect& other) nogil
        ExplicitBitVect operator&(const ExplicitBitVect& other) nogil
        ExplicitBitVect operator|(const ExplicitBitVect& other) nogil
        ExplicitBitVect operator~() nogil
        ExplicitBitVect operator+(const ExplicitBitVect& other) nogil

        bitset_t *dp_bits

cdef extern from "<boost/python/extract.hpp>" namespace "boost::python":
    cdef cppclass extract[T]:  # boost::python::extract<T>
        # Constructor from PyObject
        extract(PyObject* obj)
        # Conversion operator
        T operator()() const
        bint check() const

ctypedef extract[ExplicitBitVect*] extract_bitvec_ptr




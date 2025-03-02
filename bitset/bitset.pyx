# distutils: language = c++

from bitset cimport *
from cython.operator cimport dereference as deref
from libcpp.iterator cimport back_inserter

cdef class Bitset:
    cdef:
        bitset_t *_this

    def __cinit__(self, size_t size):
        self._this = new bitset_t(size)

    def __dealloc__(self):
        if self._this == NULL:
            del self._this

    def __repr__(self):
        cdef string s
        to_string(deref(self._this), s)
        return "<Bitset: {}>".format(s)

    def set(self, size_t i):
        self._this.set(i)

    def test(self, size_t i):
        return self._this.test(i)

    def is_subset_of(self, Bitset other):
        return self._this.is_subset_of(deref(other._this))# example.pyx

    def to_int_arr(self):
        cdef size_t num_blocks = self._this.num_blocks()
        cdef vector[unsigned int] blocks
        # cdef array128 blocks

        blocks.reserve(num_blocks)
        to_block_range(deref(self._this), back_inserter(blocks))

        # Convert to Python list
        return [<unsigned int> blocks[i] for i in range(num_blocks)]
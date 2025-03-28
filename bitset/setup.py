from distutils.core import setup
from Cython.Build import cythonize
import sys
import numpy

# HACK
sys.argv = ['setup.py', 'build_ext', '--inplace']

setup(
    name = "tutorial app",
    ext_modules = cythonize(
        '*.pyx',
    ),
)

from bitset_block_range import example
example()
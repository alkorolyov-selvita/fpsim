from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
from pathlib import Path

sys.argv = ['setup.py', 'build_ext', '--inplace']


conda_base = os.getenv('CONDA_PREFIX')
lib_path = os.path.join(conda_base, 'lib')
rdkit_include_path = os.path.join(conda_base, 'include', 'rdkit')

# RDKit paths
# rdkit_include_path = '/home/ergot/miniforge3/pkgs/rdkit-2023.09.6-py310hde493be_2/include/rdkit'
# lib_path = '/home/ergot/miniforge3/pkgs/rdkit-2023.09.6-py310hde493be_2/lib'

# Set Boost include and library paths
# boost_include = "/usr/local/include"  # Replace with your Boost include path
# boost_library_path = "/usr/local/lib"  # Replace with your Boost library path

# Define the Cython extension
extensions = [
    Extension(
        name="chemivec",
        sources=["*.pyx"],  # Replace with your actual Cython file name
        extra_compile_args=[
            '-O3',
            '-fopenmp',
        ],
        extra_link_args=[
            '-fopenmp'
        ],
        include_dirs=[
            numpy.get_include(),
            rdkit_include_path,
            # boost_include,
        ],
        library_dirs=[
            lib_path,
            # boost_library_path
        ],
        libraries=[
            "RDKitRDGeneral",
            "RDKitDataStructs",
            # "boost_numpy310",
            "boost_python310",
        ],
        language="c++",
    ),
]

# Setup
setup(
    name="chemivec",
    ext_modules=cythonize(extensions, language_level = "3str"),
    # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

from chemivec import run_tests
run_tests()
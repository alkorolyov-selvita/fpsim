# Expose specific functions/classes that you want in your public API
from .libfpsim import calc_cross_diff_float32 as calc_cross_diff
from.libfpsim import tanimoto_matrix_gpu as tanimoto_similarity_matrix
from .libfpsim import get_popcounts, bitvec_arr_to_numpy

# Define the public API
__all__ = [
    'calc_cross_diff', 'tanimoto_similarity_matrix',
    'get_popcounts', 'bitvec_arr_to_numpy'
]
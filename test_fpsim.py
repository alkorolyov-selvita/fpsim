from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

from fpsim.libfpsim import *

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


def tanimoto_matrix_rdkit(bitvecs1, bitvecs2):
    if isinstance(bitvecs1, pd.Series):
        bitvecs1 = bitvecs1.values
    if isinstance(bitvecs2, pd.Series):
        bitvecs2 = bitvecs2.values

    res = np.zeros((len(bitvecs1), len(bitvecs2)), dtype=np.float32)
    for i in range(len(bitvecs1)):
        res[i] = np.array(BulkTanimotoSimilarity(bitvecs1[i], bitvecs2))
    return res



def test_tanimoto_matrix_numpy():
    # Create random fingerprints (10 fingerprints, each with 16 uint64 elements)
    np.random.seed(42)  # For reproducibility
    n_fps = 10
    fp_length = 2048

    # Generate random fingerprints
    fps1 = np.random.randint(0, 2**64-1, size=(n_fps, fp_length), dtype=np.uint64)
    fps2 = np.random.randint(0, 2**64-1, size=(n_fps, fp_length), dtype=np.uint64)

    # Create a copy to test self-similarity
    fps1_copy = fps1.copy()

    # Calculate similarity matrices
    sim_matrix = tanimoto_matrix_numpy(fps1, fps2)
    self_sim_matrix = tanimoto_matrix_numpy(fps1, fps1_copy)

    # Test 1: Check matrix dimensions
    assert sim_matrix.shape == (n_fps, n_fps), "Similarity matrix has incorrect shape"

    # Test 2: Check self-similarity (diagonal should be all 1.0)
    assert np.all(np.isclose(np.diag(self_sim_matrix), 1.0)), "Self-similarity diagonal values are not all 1.0"

    # Test 3: Check symmetry of self-similarity matrix
    assert np.allclose(self_sim_matrix, self_sim_matrix.T), "Self-similarity matrix is not symmetric"

    # Test 4: Check range of values (all should be between 0 and 1)
    assert np.all((sim_matrix >= 0) & (sim_matrix <= 1)), "Similarity values outside the valid range [0,1]"

    # Test 5: Manually calculate and verify one similarity value
    fp1 = fps1[0]
    fp2 = fps2[0]

    # Manual calculation
    popcount1 = sum(bin(int(x)).count('1') for x in fp1)
    popcount2 = sum(bin(int(x)).count('1') for x in fp2)
    intersection = sum(bin(int(x & y)).count('1') for x, y in zip(fp1, fp2))
    expected_sim = intersection / (popcount1 + popcount2 - intersection)

    assert np.isclose(sim_matrix[0, 0], expected_sim), f"Manual calculation {expected_sim} doesn't match matrix value {sim_matrix[0, 0]}"

    # Test 6: Check real fingerprints
    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'])
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))

    fps_np = bitvec_arr_to_numpy(fps)
    sim_matrix = tanimoto_matrix_numpy(fps_np, fps_np)
    sim_matrix_rdkit = tanimoto_matrix_rdkit(fps, fps)
    assert np.allclose(sim_matrix, sim_matrix_rdkit), "Similarity matrix doesnt match RDKit"

    print("test_tanimoto_matrix_numpy passed")


def test_tanimoto_matrix_bitvec():
    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'])
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))

    sim_matrix = tanimoto_matrix_bitvec(fps, fps)
    sim_matrix_rdkit = tanimoto_matrix_rdkit(fps, fps)
    assert np.allclose(sim_matrix, sim_matrix_rdkit), "Similarity matrix doesnt match RDKit"

    print("test_tanimoto_matrix_bitvec passed")


def test_tanimoto_matrix_gpu():

    # Create a small, controlled test dataset
    # Using very small numbers to make manual verification easy
    fps1 = np.array([
        [0b1100, 0b0011],  # 2 bits in first, 2 bits in second = 4 total
        [0b1010, 0b0101],  # 2 bits in first, 2 bits in second = 4 total
        [0b1111, 0b0000]  # 4 bits in first, 0 bits in second = 4 total
    ], dtype=np.uint64)

    fps2 = np.array([
        [0b1010, 0b1010],  # 2 bits in first, 2 bits in second = 4 total
        [0b0000, 0b1111]  # 0 bits in first, 4 bits in second = 4 total
    ], dtype=np.uint64)

    # Calculate population counts manually for clarity
    popcnts1 = np.array([4, 4, 4], dtype=np.uint32)
    popcnts2 = np.array([4, 4], dtype=np.uint32)

    # Expected results (manually calculated)
    # For fps1[0] and fps2[0]:
    #   common bits = 1 (from first word) + 1 (from second word) = 2
    #   union bits = 4 + 4 - 2 = 6
    #   similarity = 2/6 = 0.333...
    # For fps1[0] and fps2[1]:
    #   common bits = 0 (from first word) + 2 (from second word) = 2
    #   union bits = 4 + 4 - 2 = 6
    #   similarity = 2/6 = 0.333...
    # And so on...
    expected = np.array([
        [2 / 6, 2 / 6],  # fps1[0] with fps2[0] and fps2[1]
        [2 / 6, 2 / 6],  # fps1[1] with fps2[0] and fps2[1]
        [2 / 6, 0 / 8]  # fps1[2] with fps2[0] and fps2[1]
    ], dtype=np.float32)

    # Calculate using CPU implementation
    cpu_result = tanimoto_matrix_cpu(fps1, fps2, popcnts1, popcnts2)

    # Calculate using GPU implementation
    gpu_result = tanimoto_matrix_gpu(fps1, fps2, popcnts1, popcnts2)

    # Assert results match within tolerance
    assert np.allclose(cpu_result, expected, rtol=1e-5, atol=1e-5), "CPU result doesn't match expected"
    assert np.allclose(gpu_result, expected, rtol=1e-5, atol=1e-5), "GPU result doesn't match expected"
    assert np.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5), "CPU and GPU results don't match"
    print('test_tanimoto_matrix_gpu passed')


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
    # print('Numpy array:', numpy_arr)
    # print('Data type:', numpy_arr.dtype)

    # Verify bits are set correctly
    # Bit 1 should be in the first uint64 block (block 0)
    # Bit 65 should be in the second uint64 block (block 1)
    assert numpy_arr[0] == 2  # 2^1 = 2
    assert numpy_arr[1] == 2  # 2^(65-64) = 2^1 = 2

    print('test_bitvec_to_numpy passed')


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

    # Verify bits are set correctly
    for i in range(num_vecs):
        # Verify first block (bits 0-63)
        assert numpy_arr_2d[i, 0] == (1 << i)
        # Verify second block (bits 64-127)
        assert numpy_arr_2d[i, 1] == (1 << i)

    print('test_bitvec_arr_to_numpy passed')


def test_calc_cross_diff():
    np.random.seed(42)
    a1 = np.random.random(10).astype(np.float32)
    a2 = np.random.random(20).astype(np.float32)

    assert np.allclose(calc_cross_diff_np(a1, a2), calc_cross_diff_float32(a1, a2))
    print('test_calc_cross_diff passed')


# Simple benchmark function
def benchmark_bitvec_arr_to_numpy():
    from rdkit.Chem import DataStructs

    # Function to create random bit vectors
    def create_random_bitvecs(n, size):
        """Create n random bit vectors of given size"""
        vectors = []
        for _ in range(n):
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

    timeit(bitvec_arr_to_numpy, vectors, desc='bitvec_arr_to_numpy')


def benchmark_tanimoto_matrix_numpy():
    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 800)
    smiles = smiles.sample(frac=1, random_state=42)
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))


    fps_packed = bitvec_arr_to_numpy(fps)
    timeit(
        tanimoto_matrix_numpy, fps_packed, fps_packed,
        kwargs={'n_jobs': 1}, desc='tanimoto numpy'
    )


def benchmark_tainimoto_matrix_bitvec():
    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 800)
    smiles = smiles.sample(frac=1, random_state=42)
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))
    timeit(tanimoto_matrix_bitvec, fps, fps, desc='tanimoto bitvec 4000 x 4000')


def benchmark_tanimoto_matrix_gpu():
    smiles = pd.Series(['CC', 'CO', 'CN', 'CF', 'CS'] * 800)
    smiles = smiles.sample(frac=1, random_state=42)
    mols = smiles.apply(MolFromSmiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = pd.Series(fpgen.GetFingerprints(mols.values))


    fps_np = bitvec_arr_to_numpy(fps)
    popcounts = get_popcounts(fps_np)

    timeit(
        tanimoto_matrix_gpu,
        fps_np, fps_np,
        popcounts, popcounts,
        desc='tanimoto gpu 4000 x 4000'
    )


    # timeit(
    #     tanimoto_matrix_gpu,
    #     fps_packed[:2000], np.concatenate([fps_packed] * 50),
    #     popcounts, np.concatenate([popcounts] * 50),
    #     kwargs={'batch_size': 100_000},
    #     desc='tanimoto gpu [2_000, 500_000]'
    # )

    # timeit(
    #     similarity_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 128}, desc='tanimoto gpu 128',
    # )
    # timeit(
    #     similarity_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 256}, desc='tanimoto gpu 256',
    # )
    # timeit(
    #     tanimoto_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 512}, desc='tanimoto gpu 512',
    # )
    # timeit(
    #     tanimoto_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 1024}, desc='tanimoto gpu 1024',
    # )
    # timeit(
    #     similarity_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 2048}, desc='tanimoto gpu 2048',
    # )
    # timeit(
    #     similarity_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 4096}, desc='tanimoto gpu 4096',
    # )
    # timeit(
    #     similarity_matrix_gpu, fps_packed, fps_packed, popcounts, popcounts,
    #     kwargs={'batch_size': 8192}, desc='tanimoto gpu 8192',
    # )


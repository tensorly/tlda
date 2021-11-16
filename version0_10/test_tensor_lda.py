import numpy as np
import scipy
from scipy.stats import gamma
from sklearn.decomposition import PCA, IncrementalPCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal

# Import utility functions from other files
import tensor_lda_clean as tlc
import test_util

tl.set_backend('numpy')
device = 'cpu'#cuda

def test_get_M1():
    a = tl.tensor([[1, 2, 0, 3, 0],
                   [2, 0, 1, 0, 5],
                   [1, 4, 3, 0, 2],
                   [0, 1, 2, 1, 0]])
    true_res = tl.tensor([1.0, 1.75, 1.5, 1.0, 1.75])
    res = tlc.get_M1(a)
    assert_array_almost_equal(res, true_res)

def test_get_M2():
    alpha_0 = 0.5
    a = tl.tensor([[1, 0, 2],
                   [0, 1, 0],
                   [1, 3, 0]])
    true_res = tl.tensor([[-0.22222222, 1.05555556, 0.77777778],
                          [1.05555556, 2.11111111, -0.44444444],
                          [0.77777778, -0.44444444, 0.77777778]])
    res = tlc.get_M2(a, tlc.get_M1(a), alpha_0)
    assert_array_almost_equal(res, true_res)

def test_get_M3 ():
    alpha_0 = 0.75
    a = tl.tensor([[1, 2],
                   [0, 1],
                   [1, 1]])
    true_res = tl.tensor([[[-0.70833333, -0.25],
                           [-0.25, 1.6875]],
                           [[-0.25, 1.6875],
                           [ 1.6875,-2.16666667]]])
    res = tlc.get_M3(a, tlc.get_M1(a), alpha_0)
    assert_array_almost_equal(res, true_res)

def test_whiten():
    a = tl.tensor([[2, 1, 0, 1, 5],
                   [1, 0, 3, 2, 3],
                   [0, 0, 4, 1, 1],
                   [1, 1, 1, 2, 1]])
    alpha_0 = 1.5
    k = 3
    true_res = tl.eye(k)
    M2 = tlc.get_M2(a, tlc.get_M1(a))
    W, _ = tlc.whiten(M2, k)
    res = tl.dot(tl.dot(W.transpose(), M), W)
    assert_array_almost_equal(res, true_res)



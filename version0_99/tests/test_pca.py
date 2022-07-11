from ..pca import PCA
import tensorly as tl
from tensorly.testing import assert_array_almost_equal
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot

def test_pca():
    a = tl.tensor([[2, 1, 0, 1, 5],
                  [1, 0, 3, 2, 3],
                  [0, 0, 4, 1, 1],
                 [1, 1, 1, 2, 1]])
    
    alpha_0 = 1.5
    k = 2
    batch_size_pca = 4

    true_res = tl.eye(k)
    M1 = tl.mean(a, axis=0)
    a_cent = tl.tensor(a - M1)

    p = PCA(k, alpha_0, batch_size= batch_size_pca)
    p.fit(a_cent)

    # check that WT M2 W = I
    M2 = (alpha_0 + 1)*tl.mean(batched_tensor_dot(a_cent, a_cent), axis=0)
    W = p.projection_weights_ / tl.sqrt(p.whitening_weights_)[None, :]
    res = tl.dot(tl.dot(W.T, M2), W)
    
    assert_array_almost_equal(res, true_res)

test_pca()
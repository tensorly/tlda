from ..tlda_wrapper import TLDA
import tensorly as tl
from tensorly.testing import assert_array_almost_equal
from tensorly.tenalg.core_tenalg import tensordot

def test_pca():
    a = tl.tensor([[2, 1, 0, 1, 5],
                  [1, 0, 3, 2, 3],
                  [0, 0, 4, 1, 1],
                 [1, 1, 1, 2, 1]])
    
    alpha_0 = 1.5
    k = 2
    batch_size_pca = 4
    true_res = tl.eye(k)

    # set params to 0 because they are not used in PCA
    tlda = TLDA(k, alpha_0, 0, 0, 0, pca_batch_size=batch_size_pca)

    tlda._partial_fit_first_order(a)
    tlda._partial_fit_second_order(a)

    a_cent = a - tlda.mean

    # check that WT M2 W = I
    M2 = (alpha_0 + 1)*tl.mean(tensordot(a_cent, a_cent, modes=1), axis=0)
    W = tlda.second_order.projection_weights_ / tl.sqrt(tlda.second_order.whitening_weights_)[None, :]
    res = tl.dot(tl.dot(W.T, M2), W)
    
    assert_array_almost_equal(res, true_res)

test_pca()
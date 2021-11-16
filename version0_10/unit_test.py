from test_util import * 
from tensor_lda_clean import * 
from tensorly.testing import assert_array_almost_equal
import numpy as np
import tensor_lda_util as tl_util


def test_get_M1():
    a = tl.tensor([[1, 2, 0, 3, 0],
                   [2, 0, 1, 0, 5],
                   [1, 4, 3, 0, 2],
                   [0, 1, 2, 1, 0]])
    true_res = tl.tensor([1.0, 1.75, 1.5, 1.0, 1.75])
    res = get_M1(a)
    assert_array_almost_equal(res, true_res)

def test_get_M2():
    alpha_0 = 0.5
    a = tl.tensor([[1, 0, 2],
                   [0, 1, 0],
                   [1, 3, 0]])
    true_res = tl.tensor([[-0.22222222, 1.05555556, 0.77777778],
                          [1.05555556, 2.11111111, -0.44444444],
                          [0.77777778, -0.44444444, 0.77777778]])
    res = get_M2(a, get_M1(a), alpha_0)
    assert_array_almost_equal(res, true_res)

def test_get_M3 ():
    alpha_0 = 0.75
    a = tl.tensor([[1.0, 2.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])
    true_res = tl.tensor([[[-0.70833333, -0.25],
                           [-0.25, 1.6875]],
                           [[-0.25, 1.6875],
                           [ 1.6875,-2.16666667]]])
    res = get_M3(a, get_M1(a), alpha_0)
    assert_array_almost_equal(res, true_res)

def test_whiten():
    a = tl.tensor([[1, 0, 2,1,4],
                   [0, 1, 0,5,0],
                   [1, 3, 1,2,0],
                   [0, 0, 1,1,0],
                   [2, 0, 6,1,1]])
    alpha_0 = 1.0
    k = 3
    true_res = tl.eye(k)
    M2 = get_M2(a, get_M1(a), alpha_0)



    W, _ = whiten(M2, k,True)
    res = tl.dot(tl.dot(W.transpose(), M2), W)
    assert_array_almost_equal(res, true_res)

test_get_M1()
test_get_M2()
test_get_M3()
test_whiten()

print("Moment Calculation/Whiten Test Complete!")
def test_fit():
    num_tops   = 5
    num_tweets = 1000  # 
    density    = 10
    vocab      = 3000  # 1000 
    smoothing  = 1e-5 #1e-5

    x, mu, _, alpha_0  = get_mu(num_tops, vocab, num_tweets, density)
    weights, factors   = fit(x, num_tops, alpha_0)
    factors = factors.T
    mu = tl.tensor(mu)[:, 0, :]

    permutation, RMSE  = validate_gammad(factors, mu, num_tops = num_tops,smoothing=smoothing)
    print(permutation)
    print(RMSE)


    factors[factors < 0.] = 0.
    # smooth beta
    factors *= (1. - smoothing)
    print(factors.shape)
    factors += (smoothing / factors.shape[1])
    factors /= factors.sum(axis=0)

    for i in range(num_tops):
        print(np.correlate(factors[i,:], mu[permutation[1]][i,:]))



test_fit()
### Fit Testing


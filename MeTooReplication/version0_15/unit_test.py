from test_util import * 
from tensor_lda_clean import * 
from tensorly.testing import assert_array_almost_equal
import numpy as np
import tensor_lda_util as tl_util
from tensorly.tenalg import kronecker
import tensorly as tl

import matplotlib.pyplot as plt


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

## Test Gradient

def get_err2(factor, alphas, alpha_0, M2):
    '''get reconstruction error for M2 moment'''
    M2_pred = tl.zeros((1, (M2.shape[1]*M2.shape[1])))
    for i, a in enumerate(alphas):
        M2_pred = M2_pred + (a/alpha_0)*kronecker([factor[i], factor[i]]) #expand(1, -1)
    return tl.metrics.regression.RMSE(M2_pred.reshape(M2.shape), tl.tensor(M2))

def get_err3(factor, alphas, alpha_0, M3, n, sp= False):
    '''get reconstruction error for M3 moment'''
    # leave factors whitened to compare vs whitened M3 moment
    M3_pred = tl.zeros((1, (n*n*n)))
    for i, a in enumerate(alphas):
        M3_pred = M3_pred + (a/alpha_0)*kronecker([factor[i], factor[i], factor[i]])
    return tl.metrics.regression.RMSE(M3_pred.reshape(M3.shape), tl.tensor(M3))

def correlate(a, v):
    print(np.std(a))
    a = (a - tl.mean(a)) / (np.std(a) * len(a))
    v = (v - tl.mean(v)) /  np.std(v)
    return np.correlate(a, v)

def test_fit():
    num_tops   = 5
    num_tweets = 10000  # 
    density    = 10
    vocab      = 1000  # 1000 
    smoothing  = 1e-5 #1e-5


    x, mu, _, alpha_0  = get_mu(num_tops, vocab, num_tweets, density)
    # Fit the data
    weights_sgd, factors_sgd,  weights_parafac, factors_parafac, M2, M3, W_inv = fit(x, num_tops, alpha_0,True,min_iter = 100,max_iter=1000) # returns whitened factors SGD
    
    alpha_sgd     = tl_util.calculate_alphas(weights_sgd)
    alpha_parafac = tl_util.calculate_alphas(weights_parafac)


    print("Factors:" )
    print(factors_sgd)
    print(factors_parafac)

    

    # unwhiten the factors
    factors_unwhitened_sgd     = (tl.dot(W_inv,factors_sgd )) 
    factors_unwhitened_parafac = (tl.dot(W_inv,factors_parafac )) 

    # Postprocessing

    factors_unwhitened_sgd [factors_unwhitened_sgd  < 0.] = 0.
    # smooth beta
    factors_unwhitened_sgd  *= (1. - smoothing)
    factors_unwhitened_sgd += (smoothing / factors_unwhitened_sgd.shape[1])
    factors_unwhitened_sgd /= factors_unwhitened_sgd.sum(axis=0)

    factors_unwhitened_parafac [factors_unwhitened_parafac < 0.] = 0.
    # smooth beta
    factors_unwhitened_parafac  *= (1. - smoothing)
    factors_unwhitened_parafac += (smoothing / factors_unwhitened_parafac.shape[1])
    factors_unwhitened_parafac /= factors_unwhitened_parafac.sum(axis=0)

    # Transpose
    factors_unwhitened_sgd = factors_unwhitened_sgd.T
    factors_unwhitened_parafac = factors_unwhitened_parafac.T
    mu = tl.tensor(mu)[:, 0, :]

    print("Weights:")
    print(weights_sgd)
    print(weights_parafac)

    permutation, RMSE  = validate_gammad(factors_unwhitened_sgd, mu, num_tops = num_tops,smoothing=smoothing)
    permutation_para, RMSE_para  = validate_gammad(factors_unwhitened_sgd, factors_unwhitened_parafac, num_tops = num_tops,smoothing=smoothing)

    print("Test Against Ground Truth")
    print(permutation)
    print(RMSE)
    print("Test Against Tensorly sym_parafac")
    print(permutation_para)
    print(RMSE_para)
    


    for i in range(num_tops):
        print(correlate(factors_unwhitened_sgd[i,:], mu[permutation[1]][i,:]))
        plt.scatter(factors_unwhitened_sgd[i,:], mu[permutation[1]][i,:])
        plt.savefig('scatter'+str(i)+'.pdf')  





test_fit()
### Fit Testing


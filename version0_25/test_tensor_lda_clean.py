import numpy as np
import cupy as cp
import scipy
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal
from cumulant_gradient import cumulant_gradient

#Insert Plotly
import matplotlib.pyplot as plt
import pickle
# Import utility functions from other files
from tlda_final import TLDA
from pca import PCA
import test_util
import tensor_lda_util as tl_util

backend="cupy"
tl.set_backend(backend)
device = 'cuda'#cuda

def test_whiten():
    #a = tl.tensor([[2, 1, 0, 1, 5],
    #               [1, 0, 3, 2, 3],
    #               [0, 0, 4, 1, 1],
    #              [1, 1, 1, 2, 1]])
    #alpha_0 = 1.5
    a       = pickle.load( open('synthetic_data.obj', 'rb'))
    alpha_0 = pickle.load( open('true_alpha.obj', 'rb') )
    k = 2
    batch_size_pca = 2000
    true_res = tl.eye(k)
    M1 = tl.mean(a, axis=0)
    a_cent = tl.tensor(a - M1)

    p = PCA(k, alpha_0,batch_size= batch_size_pca, backend=backend)
    p.fit(a_cent)
    M2 = (alpha_0 + 1)*tl.mean(batched_tensor_dot(a_cent, a_cent), axis=0)
    W = p.projection_weights_ / tl.sqrt(p.whitening_weights_)[None, :]
    res = tl.dot(tl.dot(W.T, M2), W)
    assert_array_almost_equal(res, true_res)

def test_gradient():
    a = tl.tensor([[1., 1., 1.],
                   [-1., 0., 2.],
                   [0., -1., -3.]])
    grad_true = tl.transpose(tl.tensor([-187., -324., -178.]))
    #fac_true = a[:, 0] + grad_true
    alpha_0 = 1
    k = 3
    lr = 1
    fac = tl.tensor([[1., 2., 1.],
                            [2., 3., 0.],
                            [1., 1., 1.]])
    grad = tl.mean(cumulant_gradient(fac, a, alpha_0), axis=0)
    grad2 = tl.zeros((3, 3))
    for i in range (len(a)):
        grad2 -= test_util.calculate_gradient(fac, a[i], alpha_0)

    assert_array_almost_equal(grad_true, (grad2/len(a))[:, 0])
    assert_array_almost_equal(grad, grad2/len(a))
    assert_array_almost_equal(grad_true, grad[:, 0])

def test_gradient2():
    a = tl.tensor([[2, 1, 0, 1, 5],
                   [1, 0, 3, 2, 3],
                   [0, 0, 4, 1, 1],
                   [1, 1, 1, 2, 1]])
    alpha_0 = 1.5
    k = 5
    lr = 0.001
    M1 = tl.mean(a, axis=0)
    a_cent = (a - M1)

    t = TLDA(5, 1.5, 100, 100, 10)
    factors = tl.copy(t.factors_)
    t.partial_fit(a_cent, lr)
    fac1 = t.factors_

    fac2 = tl.copy(factors)
    fac3 = tl.copy(factors)
    grad2 = tl.zeros(tl.shape(fac2))
    grad3 = tl.zeros(tl.shape(fac3))

    for i in range(len(a)):
        grad2 -= lr*test_util.calculate_gradient(factors, a_cent[i], alpha_0)
        grad3 -= lr*test_util.calculate_gradient_no_cent(factors, a[i], tl.mean(a, axis=0), alpha_0)

    fac2 += grad2/len(a)
    fac3 += grad3/len(a)
    print('testing gradient')
    #assert_array_almost_equal(fac2, fac3)
    assert_array_almost_equal(fac1, fac2)

def correlate(a, v):

    a = (a - tl.mean(a)) / (np.std(a) * len(a))
    v = (v - tl.mean(v)) /  np.std(v)
    return np.correlate(a, v)


def create_data():
    num_tops   = 2
    num_tweets = 2000
    density    = 15
    vocab      = 100
    smoothing  = 1e-5 #1e-5
    seed       = 1 


    '''get and whiten the data'''
    x, mu, _, alpha_0 = test_util.get_mu(num_tops, vocab, num_tweets, density, seed)
 
    pickle.dump(x, open('synthetic_data.obj', 'wb'))
    pickle.dump(alpha_0, open('true_alpha.obj', 'wb') )
    pickle.dump(mu, open('true_mu.obj', 'wb'))

def test_fit():
    num_tops = 2
    num_tweets = 2000
    density = 15
    vocab   = 100
    n_iter_train     = 2001
    batch_size_pca =  2000
    batch_size_grad  = 10
    n_iter_test = 10 
    theta_param =  10
    learning_rate = 0.01
    smoothing  = 1e-5 #1e-5

    x       = pickle.load( open('synthetic_data.obj', 'rb'))
    alpha_0 = pickle.load( open('true_alpha.obj', 'rb') )
    mu      = pickle.load( open('true_mu.obj', 'rb'))



    x_cent = tl.tensor(x - tl.mean(x, axis=0))
    pca = PCA(num_tops, alpha_0, batch_size_pca,backend)
    pca.fit(x_cent)
    W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    x_whit = pca.transform(x_cent)

    '''fit the tensor lda model'''
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,gamma_shape = 1.0, smoothing = 1e-6,theta=theta_param)
    tlda.fit(x_whit,W,verbose=True)
    factors_unwhitened = pca.reverse_transform(tlda.factors_)
    factors_unwhitened = factors_unwhitened.T
    '''Post-Processing '''

    # Postprocessing

    #This is hard-coded. We should calculate the alphas by hand. 
    wc   =  cp.asarray(tl.mean(x, axis=0))/vocab*(1/num_tops)
    wc   =  tl.reshape(wc,(vocab,1))
    
    factors_unwhitened   =  cp.asarray(factors_unwhitened)
    factors_unwhitened += wc

    #print(factors_unwhitened.dtype)
    #print(wc.dtype)
    print(factors_unwhitened.shape)
    #print(wc.shape)
 
    print(factors_unwhitened)
    factors_unwhitened [factors_unwhitened  < 0.] = 0.
    # smooth beta
    factors_unwhitened  *= (1. - smoothing)
    print(factors_unwhitened)

    factors_unwhitened += (smoothing / factors_unwhitened.shape[1])
    print(factors_unwhitened)
    print("begin print estimated mu")
    factors_unwhitened /= factors_unwhitened.sum(axis=0)
    print(factors_unwhitened)
    # remean the data
    print("begin mean")

    print(wc)
    print("begin ground truth")
    print(mu)


    '''test RMSE'''
    mu = cp.asarray(mu)[:, 0, :]
    permutation,RMSE = test_util.validate_gammad(factors_unwhitened.T, mu, num_tops = num_tops)


    print("Fit RMSE: " + str(RMSE.item()))
    print("Test Against Ground Truth")
    print(permutation)

    accuracy = []
    for i in range(num_tops):
        accuracy.append(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][i,:]))
        print(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][i,:]))
        plt.scatter(cp.asnumpy(factors_unwhitened.T[i,:]), cp.asnumpy(mu[permutation[1]][i,:]))
        plt.savefig('scatter'+str(i)+'.pdf')
        plt.clf()  
    
    plt.scatter(cp.asnumpy(factors_unwhitened.T[0,:]), cp.asnumpy(factors_unwhitened.T[1,:]))
    plt.savefig('est_map'+'.pdf')
    plt.clf()

    plt.scatter(cp.asnumpy(mu[permutation[1]][0,:]), cp.asnumpy(mu[permutation[1]][1,:]))
    plt.savefig('true_map'+'.pdf')
    plt.clf()

    #tl.mean(cp.asnumpy(accuracy))

def test_inference():
    num_tops   = 2
    num_tweets = 5000  # 
    density    = 5
    vocab      = 1000  # 1000 
    smoothing  = 1e-5 #1e-5
    '''get and whiten the data'''
    x, _, theta, alpha_0 = test_util.get_mu(num_tops, vocab, num_tweets, density)
    x_cent = x - tl.mean(x, axis=0)
    pca = PCA(num_tops, alpha_0, num_tweets//1)
    pca.fit(x_cent)
    x_whit = pca.transform(x_cent)

    '''fit the tensor lda model'''
    tlda = TLDA(num_tops, alpha_0, 300, 300, 100)
    tlda.fit(x_whit,W,verbose=True)
    factors = pca.reverse_transform(tlda.factors_)
    tlda.factors_ = factors

    '''infer the document-topic distribution and test RMSE'''
    doc_topic_dist, topic_word_dist = tlda.predict(x, parallel = False)
    doc_topic_dist, topic_word_dist = tlda.predict(x)
    _, RMSE, rec_err = test_util.validate_gammad(doc_topic_dist, theta, transpose=True, num_tops = num_tops)
    print("Inference RMSE: " + str(RMSE.item()))

    assert(RMSE.item() < 0.3)

def main():
    #print("Begin Data Creation")
    #create_data()
    #print("End Data Creation")
    #tl.set_backend(backend)
    print("Begin Whitening")
    test_whiten()
    print("End Whitening")
    #test_gradient2()
    print("Begin Fit")
    test_fit()
    print("End Fit")
    #test_inference()
    print("Done!")

    return

if __name__ == '__main__':
    main()

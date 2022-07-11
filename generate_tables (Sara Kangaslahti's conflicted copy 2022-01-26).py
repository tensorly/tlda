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
# from cumulant_gradient import cumulant_gradient

import time
import csv
import random
import sys

#Insert Plotly
import matplotlib.pyplot as plt
import pickle
# Import utility functions from other files
from version0_20.tlda_final import TLDA
from version0_20.pca import PCA
import version0_20.test_util as test_util
import version0_20.tensor_lda_util as tl_util

import version0_15.tensor_lda_clean as tlda_mid

from sklearn.decomposition import LatentDirichletAllocation as sklearn_LDA

backend="cupy"
tl.set_backend(backend)

VOCAB = 500 # 1000

def loss_rec(factor, cumulant, theta):
    # cumulant = M3 - compute this
    rec = tl.cp_to_tensor((None, [factor]*3))
    #rec_loss = tl.tenalg.inner(rec, cumulant)
    rec_loss = tl.norm(rec - cumulant, 2)**2
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2
    return ortho_loss + rec_loss, ortho_loss, rec_loss#/tl.norm(cumulant, 2)

def correlate(a, v):
    a = cp.asarray((a - tl.mean(a)) / (np.std(a) * len(a)))
    v = cp.asarray((v - tl.mean(v)) /  np.std(v))
    return np.correlate(a, v)

def create_data(vocab= 500, seed= None):
    num_tops   = 2
    num_tweets = 20000 #20000
    density    = 15
    vocab      = vocab # 100
    smoothing  = 0.001 #1e-5 #1e-5
    seed       = None

    print("Vocab: " + str(vocab))
    print("num_tweets: " + str(num_tweets))
    print("density: " + str(density))

    '''get and whiten the data'''
    x = None
    while x is None:
        try:
            x, mu, _, alpha_0 = test_util.get_mu(num_tops, vocab, num_tweets, density, seed)
        except ValueError:
            pass
 
    pickle.dump(x, open('data/synthetic_data.obj', 'wb'))
    pickle.dump(alpha_0, open('data/true_alpha.obj', 'wb') )
    pickle.dump(mu, open('data/true_mu.obj', 'wb'))

def postprocess(factors_unwhitened, mu, x, vocab, num_tops, smoothing, decenter=False, name="", alpha_0 = 1):
    '''Post-Processing '''
    res = []
    # Postprocessing
    if decenter == True:
        plt.scatter(cp.asnumpy(factors_unwhitened.T[0,:]), cp.asnumpy(factors_unwhitened.T[1,:]))
        plt.savefig('results/est_map_no_postprocessing'+name+'.pdf')
        plt.clf()

        plt.scatter(cp.asnumpy(factors_unwhitened.T[0,:]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/est_map_0_vs_m1_no_postprocess'+name+'.pdf')
        plt.clf()

        plt.scatter(cp.asnumpy(factors_unwhitened.T[1,:]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/est_map_1_vs_m1_no_postprocess'+name+'.pdf')
        plt.clf()

        plt.scatter(cp.asnumpy(mu[0]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/mu_0_vs_m1'+name+'.pdf')
        plt.clf()

        plt.scatter(cp.asnumpy(mu[1]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/mu_1_vs_m1'+name+'.pdf')
        plt.clf()

    #This is hard-coded. We should calculate the alphas by hand. 
    if decenter == True:
        factors_no_M1 = tl.copy(factors_unwhitened)
        t1 = time.time()
        eig_vals = cp.array([np.linalg.norm(k)**3 for k in factors_unwhitened.T ])
        # normalize beta
        alpha           = cp.power(eig_vals, -2)
        alpha_norm      = (alpha / alpha.sum()) * alpha_0
        weights   = tl.tensor(alpha_norm)
        # print("weights shape:")
        # print(weights.shape)
        scale_weights = weights**(1/3)

        fac2 = factors_unwhitened/scale_weights
        # print("fac2 shape: ")
        # print(fac2.shape)
        fac2 = (fac2.T + tl.mean(x, axis=0)).T
        fac2 *= scale_weights
        fac2 = cp.asarray(fac2)
        t2 = time.time()
        res.append((name + ' decentering', t2-t1))
        # print("final fac2: ")
        # print(fac2)


        print("decenter with new strategy:")
        print(fac2[0])
        t1 = time.time()
        wc   =  cp.asarray(tl.mean(x, axis=0))# /vocab*(1/num_tops)
        wc   =  tl.reshape(wc,(vocab,1))
        
        factors_unwhitened   =  cp.asarray(factors_unwhitened)
        factors_unwhitened += wc
        t2 = time.time()
        # print("Decentering: " + str(t2-t1))
        # res.append((name + ' decentering', t2-t1))
        print("decenter with old strategy:")
        print(factors_unwhitened[0])

        plt.scatter(cp.asnumpy(factors_unwhitened.T[0,:]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/est_map_0_vs_m1'+name+'.pdf')
        plt.clf()

        plt.scatter(cp.asnumpy(factors_unwhitened.T[1,:]), cp.asnumpy(tl.mean(x, axis=0)))
        plt.savefig('results/est_map_1_vs_m1'+name+'.pdf')
        plt.clf()

    #print(factors_unwhitened.dtype)
    #print(wc.dtype)
    #print(factors_unwhitened.shape)
    #print(wc.shape)
    if decenter == True:
        # adjusted_factors = tl_util.non_negative_adjustment(factors_unwhitened)
        # adjusted_factors = tl_util.smooth_beta(adjusted_factors, smoothing=smoothing)
        # adjusted_factors /= adjusted_factors.sum(axis=0)
        # plt.scatter(cp.asnumpy(adjusted_factors.T[0,:]), cp.asnumpy(adjusted_factors.T[1,:]))
        # plt.savefig('results/est_map_adjusted_postprocessing'+name+'.pdf')
        # plt.clf()

        factors_no_M1   =  cp.asarray(factors_no_M1)
        factors_no_M1[factors_no_M1  < 0.] = 0.
        factors_no_M1 *= (1. - smoothing)
        factors_no_M1 += (smoothing / factors_no_M1.shape[1])
        factors_no_M1 /= factors_no_M1.sum(axis=0)




    factors_unwhitened   =  cp.asarray(factors_unwhitened)
    # print(factors_unwhitened)
    t1 = time.time()
    factors_unwhitened [factors_unwhitened  < 0.] = 0.
    # smooth beta
    factors_unwhitened  *= (1. - smoothing)
    #print(factors_unwhitened)

    factors_unwhitened += (smoothing / factors_unwhitened.shape[1])
    #print(factors_unwhitened)
    #print("begin print estimated mu")
    factors_unwhitened /= factors_unwhitened.sum(axis=0)
    t2 = time.time()
    if decenter == False:
        print("Smoothing and Normalization: " + str(t2-t1))
        res.append((name + ' smoothing and normalization', t2-t1))
    #print(factors_unwhitened)
    # remean the data
    #print("begin mean")
    if decenter == True:
        t1 = time.time()
        fac2[fac2 < 0.] = 0.
        fac2 *= (1. - smoothing)
        fac2 += (smoothing/fac2.shape[1])
        fac2 /= fac2.sum(axis=0)
        t2 = time.time()
        res.append((name + ' smoothing and normalization', t2-t1))
    # print(wc)
    # print("begin ground truth")
    # print(mu)


    '''test RMSE'''
    mu = np.asarray(mu)[:, 0, :]
    permutation,RMSE = test_util.validate_gammad(factors_unwhitened.T, mu, num_tops = num_tops)
    if decenter==True:
        permutation_fac2, RMSE2 = test_util.validate_gammad(fac2.T, mu, num_tops = num_tops)
        print("Fit RMSE new decenter: " + str(RMSE2.item()))
    print("Fit RMSE: " + str(RMSE.item()))
    print(name + " Test Against Ground Truth")
    # print(permutation)

    outFile = open("results/accuracies"+name+".txt", 'w')

    print(mu.shape, file=outFile)
    print(mu[permutation[1]].shape, file=outFile)

    accuracy = []
    for i in range(num_tops):
        if decenter == False:
            accuracy.append(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][i,:]))
        print(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][i,:]), file=outFile)
        # if name=="parafac":
        #     print("Reverse permutation: ")
        #     print(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][(i+1)%2,:]))
        plt.scatter(cp.asnumpy(factors_unwhitened.T[i,:]), cp.asnumpy(mu[permutation[1]][i,:]))
        if decenter == False:
            plt.savefig('results/scatter'+str(i)+'.pdf')
        else:
            plt.savefig('results/scatter'+str(i)+'_addM1.pdf')
        plt.clf()  

    if decenter == True:
        accuracy_fwd = []
        accuracy_rev = []
        for i in range(num_tops):
            accuracy.append(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][i,:]))
            #accuracy.append((correlate(fac2.T[i,:], mu[permutation_fac2[1]][i,:])))

            # testing reverse direction
            # accuracy_rev.append((correlate(fac2.T[i,:], mu[permutation_fac2[1]][(i+1)%2,:])))
            print(correlate(fac2.T[i,:], mu[permutation_fac2[1]][i,:]), file=outFile)
            # print((correlate(fac2.T[i,:], mu[permutation_fac2[1]][(i+1)%2,:])))
            # if name=="parafac":
            #     print("Reverse permutation: ")
            #     print(correlate(factors_unwhitened.T[i,:], mu[permutation[1]][(i+1)%2,:]))
        # if sum(accuracy_fwd) >= sum(accuracy_rev):
        #     accuracy = accuracy_fwd
        # else:
        #     accuracy = accuracy_rev
        #     permutation_fac2 = (permutation_fac2[0], [(x + 1)%2 for x in permutation_fac2[1]])
        # for i in range(num_tops):
        #     print(correlate(adjusted_factors.T[i,:], mu[permutation_fac2[1]][i,:]), file=outFile)
        for i in range(num_tops):
            print(correlate(factors_no_M1.T[i,:], mu[permutation_fac2[1]][i,:]), file=outFile)
        for i in range(num_tops):
            plt.scatter(cp.asnumpy(fac2.T[i,:]), cp.asnumpy(mu[permutation_fac2[1]][i,:]))
            plt.savefig('results/scatter_2_'+str(i)+'.pdf')
            plt.clf() 
    outFile.close()
    if decenter == False:
        plt.scatter(cp.asnumpy(factors_unwhitened.T[0,:]), cp.asnumpy(factors_unwhitened.T[1,:]))
    else:
        plt.scatter(cp.asnumpy(fac2.T[0,:]), cp.asnumpy(fac2.T[1,:]))
    plt.savefig('results/est_map'+name+'.pdf')
    plt.clf()

    if decenter == False:
        plt.scatter(cp.asnumpy(mu[permutation[1]][0,:]), cp.asnumpy(mu[permutation[1]][1,:]))
    else:
        plt.scatter(cp.asnumpy(mu[permutation_fac2[1]][0,:]), cp.asnumpy(mu[permutation_fac2[1]][1,:]))
    plt.savefig('results/true_map'+name+'.pdf')
    plt.clf()
    return res, accuracy

def gen_fit_0_15(n_iter_max=2000):
    num_tops = 2
    vocab   = VOCAB
    smoothing  =  0.001 #1e-5 #1e-5

    res = []
    x       = pickle.load( open('data/synthetic_data.obj', 'rb'))
    alpha_0 = pickle.load( open('data/true_alpha.obj', 'rb') )
    mu      = pickle.load( open('data/true_mu.obj', 'rb'))

    backend="cupy"
    tl.set_backend(backend)

    x = tl.tensor(x)

    t1 = time.time()
    M1 = tlda_mid.get_M1(x)
    t2 = time.time()
    print("M1: " + str(t2-t1))
    res.append(('M1 calc', t2-t1))

    t1 = time.time()
    M2_img = tlda_mid.get_M2(x, M1, alpha_0)
    t2 = time.time()
    print("M2: " + str(t2-t1))
    res.append(('M2 calc', t2-t1))

    t1 = time.time()
    W, W_inv = tlda_mid.whiten(M2_img, num_tops) # W (n_words x n_topics)
    t2 = time.time()
    print(tl.dot(tl.dot(W.T, M2_img), W))
    print("W: " + str(t2-t1))
    res.append(('W calc', t2-t1))

    W = tl.tensor(W)
    W_inv = tl.tensor(W_inv)

    t1 = time.time()
    X_whitened = tl.dot(x, W)
    t2 = time.time()
    print("Whiten X: " + str(t2-t1))
    res.append(('whiten X', t2-t1))

    res_copy = res.copy()

    # This is where the two versions branch off -- begin with version 0.10
    t1 = time.time()
    M1_whitened = tl.dot(M1, W)
    t2 = time.time()
    print("Whiten M1: " + str(t2-t1))
    res.append(('whiten M1', t2-t1))

    t1 = time.time()
    M3_final = tlda_mid.get_M3(X_whitened, M1_whitened, alpha_0)
    t2 = time.time()
    print("Parafac M3: " + str(t2-t1))
    res.append(('construct M3', t2-t1))

    t1 = time.time()
    lambdas_learned_parafac, phis_learned_parafac = sym_parafac(M3_final, rank=num_tops, n_repeat=100, n_iteration=1000, verbose=False)
    t2 = time.time()
    print("Parafac Decomposition: " + str(t2-t1))
    res.append(('decompose parafac', t2-t1))

    t1 = time.time()
    factors_unwhitened_parafac     = (tl.dot(W_inv,phis_learned_parafac )) 
    t2 = time.time()
    print("Unwhitening parafac factors: " + str(t2-t1))
    res.append(('unwhiten factors parafac', t2-t1))

    t1 = time.time()
    weights, phis_learned  = tlda_mid.simulate_all(X_whitened, alpha_0, num_tops, lr1 = 0.001, verbose = False,min_iter = 100,max_iter=n_iter_max)
    t2 = time.time()
    print("SGD Calc: " + str(t2-t1))
    res_copy.append(('SGD calc', t2-t1))

    t1 = time.time()
    factors_unwhitened     = (tl.dot(W_inv,phis_learned )) 
    t2 = time.time()
    print("Unwhitening factors: " + str(t2-t1))
    res_copy.append(('unwhiten factors SGD', t2-t1))

    res3, accuracy_parafac = postprocess(factors_unwhitened_parafac, mu, x, vocab, num_tops, smoothing, decenter=False, name="parafac")
    # res3 = {}
    # res2 = []
    # accuracy_uncentered = None
    res2, accuracy_uncentered = postprocess(factors_unwhitened, mu, x, vocab, num_tops, smoothing, decenter=False)
    # {**{**res, **res2}, **res3}
    res.extend(res3)
    res_copy.extend(res2)
    return res, res_copy, accuracy_parafac, accuracy_uncentered

def gen_fit_0_20(n_iter_train = 2001, batch_size_grad= 100, vocab = VOCAB, theta=1, learning_rate = 0.01, seed=None):
    num_tops = 2
    vocab   = vocab
    n_iter_train     = n_iter_train
    batch_size_pca =  20000 # 2000
    batch_size_grad  = batch_size_grad # 10
    n_iter_test = 10 
    theta_param = theta #1 #1 # 0.5 # 0.5 # 50
    learning_rate = learning_rate# 0.00001
    smoothing  = 1e-5 #1e-5

    # res = {}
    res = []
    x       = pickle.load( open('data/synthetic_data.obj', 'rb'))
    alpha_0 = pickle.load( open('data/true_alpha.obj', 'rb') )
    mu      = pickle.load( open('data/true_mu.obj', 'rb'))

    backend="cupy"
    tl.set_backend(backend)
    
    x = tl.tensor(x)


    t1 = time.time()
    x_cent = tl.tensor(x - tl.mean(x, axis=0))
    t2 = time.time()
    print("Centering time: " + str(t2-t1))
    res.append(('centering', t2-t1))

    t1 = time.time()
    pca = PCA(num_tops, alpha_0, batch_size_pca,backend)
    pca.fit(x_cent)
    t2 = time.time()
    print("PCA fit: " + str(t2-t1))
    res.append(('PCA fit', t2-t1))

    # M2_img = tlda_mid.get_M2(x_cent, tl.mean(x_cent, axis=0), alpha_0)
    n = x_cent.shape[1]
    M2 = tl.zeros((1,n**2))
    ns = x_cent.shape[0]
    #diag_2 = torch.zeros(ns, n*n)
    for i in range (ns):
        #first term
        M2 += tl.tenalg.kronecker([x_cent[i,:],x_cent[i,:]]) # expand(1, -1)
    M2 /= ns
    M2 = tl.reshape(M2, (n, n))
    W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    whitening_test = tl.dot(tl.dot(W.T, M2), W)
    outFile = open("results/whitening_matrix_accuracy.txt", 'w')
    print(str(whitening_test), file=outFile)
    outFile.close()
    print(whitening_test)


    t1 = time.time()
    # W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    x_whit = pca.transform(x_cent)
    t2 = time.time()
    print("PCA Transform: " + str(t2-t1))
    res.append(('PCA transform', t2-t1))

    outFile = open("results/mu_whitened.txt", 'w')
    mu_cent = tl.tensor(cp.asarray(mu)[:, 0, :]) - tl.mean(x, axis=0)
    mu_whit_no_cent = pca.transform(tl.tensor(cp.asarray(mu)[:, 0, :])).T
    print(mu_whit_no_cent, file=outFile)
    mu_whit = (pca.transform(mu_cent)).T
    print(mu_whit, file=outFile)
    outFile.close()

    
    M3 = tlda_mid.get_M3(x_whit, tl.zeros(tl.shape(tl.mean(x_whit, axis=0))), alpha_0)

    '''fit the tensor lda model'''
    t1 = time.time()
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,cumulant = M3, gamma_shape = 1.0, smoothing = smoothing,theta=theta_param, seed=seed)
    tlda.fit(x_whit,verbose=False)
    t2 = time.time()
    print("TLDA fit: " + str(t2-t1))
    res.append(('TLDA fit', t2-t1))

    print("Whitened factor: ")
    print(tlda.factors_)
    # cumulant = tlda_mid.get_M3(x_whit, tl.mean(x, axis=0), alpha_0)
    # print("total loss, ortho loss, rec loss")
    # print(loss_rec(tlda.factors_, cumulant, theta_param))
    outFile = open("results/factors_tlda_whitened.txt", 'w')
    print(str(tlda.factors_), file=outFile)
    outFile.close()

    t1 = time.time()
    factors_unwhitened = pca.reverse_transform(tlda.factors_.T)
    factors_unwhitened = factors_unwhitened.T
    t2 = time.time()
    print("PCA Reverse Transform: " + str(t2-t1))
    res.append(('unwhiten factors', t2-t1))
    
    res2, accuracy = postprocess(factors_unwhitened, mu, x, vocab, num_tops, smoothing, True, alpha_0 = alpha_0)
    #tl.mean(cp.asnumpy(accuracy))
    # {**res, **res2}
    res.extend(res2)

    t1 = time.time()
    lda = sklearn_LDA(n_components = num_tops, doc_topic_prior = alpha_0, random_state=seed)
    lda.fit(cp.asnumpy(x))
    factors_sklearn = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    t2 = time.time()
    # outFile = open("results/factors_sklearn_whitened.txt", 'w')
    # print(str(np.dot(cp.asnumpy(factors_sklearn))), file=outFile)
    # outFile.close()
    res3 = []
    res3.append(('sklearn LDA', t2 - t1))
    print(res2)
    res4, accuracy2 = postprocess(factors_sklearn.T, mu, x, vocab, num_tops, smoothing, False, alpha_0 = alpha_0, name="sklearn")
    res3.extend(res4)
    return res, accuracy, res3, accuracy2


def main():
    print("new version")
    nums = 10 
    tot_parafac = {}
    tot_uncentered = {}
    tot_centered = {}
    tot_sklearn = {}
    acc_parafac = [] 
    acc_uncentered = []
    acc_centered = []
    acc_sklearn = []
    vocab_arr = [100, 500] # , 1000] # , 1500] # , 2000, 2500]
    theta_arr = [10000] #1]#, 5, 10] #, 25] # , 50]
    # theta_arr = [1]
    # lr_arr = [1e-2, 1e-3, 1e-4, 1e-5]# 1e-6, 1e-8]
    lr_arr = [1e-3]#[1e-3]
    seed_arr = []
    for i in range(nums*len(vocab_arr)):
        seed = random.randrange(2**32)
        seed_arr.append(seed)
    outFile = open("data/seeds.txt", 'w')
    print(str(seed_arr), file=outFile)
    outFile.close()
    j = 0

    for i in range(0, nums):
        for vocab in vocab_arr:
            create_data(vocab=vocab, seed=seed_arr[j])
            backend="cupy"
            tl.set_backend(backend)
            for lr in lr_arr:
                for theta in theta_arr:
                    # res_centered, accuracy_centered = gen_fit_0_20(n_iter_train = 40001, batch_size_grad = 50) #20
                    res_centered, accuracy_centered, res_sklearn, accuracy_sklearn = gen_fit_0_20(n_iter_train = 40001, batch_size_grad = 20, vocab=vocab, seed=seed_arr[j], theta = theta, learning_rate = lr) 
                    outFile = open('results/res2.txt', 'w')
                    print("theta = " + str(theta) + "lr = " + str(lr) + '\n' + str(res_centered) + '\n' + str(accuracy_centered), file=outFile)
                    print(str(res_sklearn) + '\n' + str(accuracy_sklearn), file=outFile)
                    outFile.close()
                    # backend="cupy"
                    # tl.set_backend(backend)
                    # res_parafac, res_uncentered, accuracy_parafac, accuracy_uncentered = gen_fit_0_15()
                    # print(res_parafac, res_uncentered, accuracy_parafac, accuracy_uncentered)
                    # res_centered, accuracy_centered = gen_fit_0_20(n_iter_train = 10001) 
                    # print(res_centered, accuracy_centered)
                    # acc_parafac.append(accuracy_parafac)
                    acc_centered.append(accuracy_centered)
                    acc_sklearn.append(accuracy_sklearn)
                    # acc_uncentered.append(accuracy_uncentered)
                    # if i == 0:
                    #     # tot_parafac = res_parafac
                    #     # tot_uncentered = res_uncentered
                    #     tot_centered[vocab] = res_centered
                    #     tot_sklearn[vocab] = res_sklearn
                    # else:
                    #     # tot_parafac = [(x, y + res_parafac[i][1]) for i, (x, y) in enumerate(tot_parafac)]
                    #     # tot_uncentered = [(x, y + res_uncentered[i][1]) for i, (x, y) in enumerate(tot_uncentered)]
                    #     tot_centered[vocab] = [(x, y + res_centered[i][1]) for i, (x, y) in enumerate(tot_centered[vocab])]
                    #     tot_sklearn[vocab] = [(x, y + res_sklearn[i][1]) for i, (x, y) in enumerate(tot_sklearn[vocab])]
            j += 1
        if i == 4 or i == 6:
            with open('results/results_correlation_20k_'+str(i)+'.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Iteration", "Vocabulary Size", "Correlation Sklearn 1", "Correlation Sklearn 2", "Correlation Centered 1", "Correlation Centered 2"])
                k = 0
                for idx in range (0, i+1):
                    for vocab1 in vocab_arr:
                        for lr1 in lr_arr:
                            for theta1 in theta_arr:
                                csvwriter.writerow([str(idx), str(vocab1), str(lr1), str(theta1), str(acc_sklearn[k][0]), str(acc_sklearn[k][1]), str(acc_centered[k][0]), str(acc_centered[k][1])])
                                k += 1
            
    # with open('results/results_20k.csv', 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["Vocab", "Step", "Time (s)"])
    #     for vocab in vocab_arr:
    #         for result_arr in [tot_sklearn[vocab], tot_centered[vocab]]: 
    #             for row in result_arr:
    #                 csvwriter.writerow([vocab, row[0], row[1]])
    
    with open('results/results_correlation_20k.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Vocabulary Size", "Learning Rate", "Theta", "Correlation Sklearn 1", "Correlation Sklearn 2", "Correlation Centered 1", "Correlation Centered 2"])
        j = 0
        for i in range (0, nums):
            for vocab in vocab_arr:
                for lr in lr_arr:
                    for theta in theta_arr:
                        csvwriter.writerow([str(i), str(vocab), str(lr), str(theta), str(acc_sklearn[j][0]), str(acc_sklearn[j][1]), str(acc_centered[j][0]), str(acc_centered[j][1])])
                        j += 1

    

    print("Done!")
    return

if __name__ == '__main__':
    main()

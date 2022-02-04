import numpy as np
import cupy as cp
import random
import numpy.random

from scipy.optimize import linear_sum_assignment
from scipy.stats import gamma
import scipy

# Import TensorLy
import tensorly as tl
from tensorly.metrics.regression import RMSE
from tensorly import norm
from tlda_final_validation import *
 
tl.set_backend('numpy')
device = 'cpu'#cuda

def get_mu(top_n, vocab_size, doc_num, t_per_doc,seed):
    tl.set_backend('numpy')
    '''use code here:
    http://www.hongliangjie.com/2010/09/30/generate-synthetic-data-for-lda/
    to get document, topic matrices'''
    ## define some constant
    np.random.seed(seed=seed)
    TOPIC_N = top_n
    VOCABULARY_SIZE = vocab_size
    DOC_NUM = doc_num
    TERM_PER_DOC = t_per_doc
    w_arr = tl.zeros((DOC_NUM, VOCABULARY_SIZE), dtype=tl.float32)

    #beta = [0.01 for i in range(VOCABULARY_SIZE)]
    #alpha = [0.9 for i in range(TOPIC_N)]
    beta = [0.01 for i in range(VOCABULARY_SIZE)]
    alpha = [0.01 for i in range(TOPIC_N)]

    mu = []
    theta_arr = np.zeros((DOC_NUM, TOPIC_N))
    ## generate multinomial distribution over words for each topic
    for i in range(TOPIC_N):
    	topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
    	mu.append(topic/np.sum(topic))
    print([np.sum(mu_i) for mu_i in mu])

    for i in range(DOC_NUM):
    	buffer = {}
    	z_buffer = {} ## keep track the true z
    	## first sample theta
    	theta = numpy.random.mtrand.dirichlet(alpha,size = 1)
    	theta /= np.sum(theta)
    	for j in range(TERM_PER_DOC):
    		## first sample z
    		z = numpy.random.multinomial(1,theta[0],size = 1)
    		z_assignment = 0
    		for k in range(TOPIC_N):
    			if z[0][k] == 1:
    				break
    			z_assignment += 1
    		if not z_assignment in z_buffer:
    			z_buffer[z_assignment] = 0
    		z_buffer[z_assignment] = z_buffer[z_assignment] + 1
    		## sample a word from topic z
    		w = numpy.random.multinomial(1,mu[z_assignment][0],size = 1)
    		w_assignment = 0
    		for k in range(VOCABULARY_SIZE):
    			if w[0][k] == 1:
    				break
    			w_assignment += 1
    		if not w_assignment in buffer:
    			buffer[w_assignment] = 0
    		buffer[w_assignment] = buffer[w_assignment] + 1
    		w_arr[i] = w_arr[i] + w
    	theta_arr[i] = theta
    return tl.tensor(w_arr), mu, theta_arr, sum(alpha)

def validate_gammad (gammad_arr, theta_arr, transpose = False, num_tops=3,smoothing=1e-6):
    tl.set_backend('numpy')

    '''get RMSE for topic distribution using heuristic'''
    factor = tl.tensor(cp.asnumpy(gammad_arr))

    factor[factor < 0.] = 0.
    # smooth beta
    factor *= (1. - smoothing)
    factor += (smoothing / factor.shape[1])
    factor /= factor.sum(axis=0)


    factor =  (factor.transpose(0, 1) / tl.norm(factor, axis=1)[:, None]).T
    factor[np.isnan(factor)] = 0
    sample = tl.tensor(cp.asnumpy(theta_arr))
    sample = (sample.transpose(0, 1) / tl.norm(sample, axis=1)[:, None]).T
    sample[np.isnan(sample)] = 0

    if transpose == False:
        tl.set_backend('numpy')
        M_corr = tl.dot(factor.T, sample)
    else:
        tl.set_backend('numpy')
        M_corr = tl.dot(factor, sample.T)
    permutation = linear_sum_assignment(-M_corr)

    if (transpose == True):
        sample = sample.T
        return permutation, tl.metrics.regression.RMSE(tl.tensor(np.array([theta_arr[:, permutation[1][i]] for i in range(num_tops)])), tl.tensor(gammad_arr.T))
    return permutation, tl.metrics.regression.RMSE(tl.tensor(np.array([sample[:, permutation[1][i]] for i in range(num_tops)])), factor.T)

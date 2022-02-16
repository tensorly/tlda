import numpy as np
import cupy as cp
import tensorly as tl
if(tl.get_backend() == "cuda"):
	from cupyx.scipy.special import digamma, gammaln
else:
	from scipy.special import  digamma, gammaln

# import sparse


# Import TensorLy
import tensorly as tl
from tensorly import norm
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot

tl.set_backend('numpy')
device = 'cpu'#cuda

def get_ei(length, i):
    '''Get the ith standard basis vector of a given length'''
    e = tl.zeros(length)
    e[i] = 1
    return e

def dirichlet_expectation(alpha):
    '''Normalize alpha using the dirichlet distribution'''
    return digamma(alpha) - digamma(sum(alpha))

def smooth_beta(beta, smoothing = 0.01):
    '''Smooth the existing beta so that it all positive (no 0 elements)'''
    smoothed_beta = beta * (1 - smoothing)
    smoothed_beta += (np.ones((beta.shape[0], beta.shape[1])) * (smoothing/beta.shape[0]))

    assert np.all(abs(np.sum(smoothed_beta, axis=0) - 1) <= 1e-6), 'sum not close to 1'
    assert smoothing <= 1e-4 or np.all(smoothed_beta > 1e-10), 'zero values'
    return smoothed_beta

def simplex_proj(V):
    '''Project V onto a simplex'''
    v_len = V.size
    U = np.sort(V)[::-1]
    cums = np.cumsum(U, dtype=float) - 1
    index = np.reciprocal(np.arange(1, v_len+1, dtype=float))
    inter_vec = cums * index
    to_befind_max = U - inter_vec
    max_idx = 0

    for i in range(0, v_len):
        if (to_befind_max[v_len-i-1] > 0):
            max_idx = v_len-i-1
            break
    theta = inter_vec[max_idx]
    p_norm = V - theta
    p_norm[p_norm < 0.0] = 0.0
    return (p_norm, theta)


def non_negative_adjustment(M):
    '''Adjust M so that it is not negative by projecting it onto a simplex'''
    M_on_simplex = np.zeros(M.shape)
    M = tl.to_numpy(M)

    for i in range(0, M.shape[1]):
        projected_vector, theta = simplex_proj(M[:, i] - np.amin(M[:, i]))
        projected_vector_revsign, theta_revsign = simplex_proj(-1*M[:, i] - np.amin(-1*M[:, i]))

        if (theta < theta_revsign):
            M_on_simplex[:, i] = projected_vector
        else:
            M_on_simplex[:, i] = projected_vector_revsign
    return M_on_simplex

def perplexity (documents, beta, alpha, gamma):
    '''get perplexity of model, given word count matrix (documents)
    topic/word distribution (beta), weights (alpha), and document/topic
    distribution (gamma)'''

    elogbeta = np.log(beta)

    corpus_part = np.zeros(documents.shape[0])
    for i, doc in enumerate(documents):
        doc_bound = 0.0
        gammad = gamma[i]
        elogthetad = dirichlet_expectation(gammad)

        for idx in np.nonzero(doc)[0]:
            doc_bound += doc[idx] * log_sum_exp(elogthetad + elogbeta[idx].T)

        doc_bound += np.sum((alpha - gammad) * elogthetad)
        doc_bound += np.sum(gammaln(gammad) - gammaln(alpha))
        doc_bound += gammaln(np.sum(alpha)) - gammaln(np.sum(gammad))

        corpus_part[i] = doc_bound

    #sum the log likelihood of all the documents to get total log likelihood
    log_likelihood = np.sum(corpus_part)
    total_words = np.sum(documents)

    #perplexity is - log likelihood / total number of words in corpus
    return (-1*log_likelihood / total_words)



def log_sum_exp(x):
    '''calculate log(sum(exp(x)))'''
    a = tl.max(x)
    
    if(tl.get_backend() == "cuda"):
        return a + cp.log(cp.sum(cp.exp(x - a)))
    else:
        return a + np.log(np.sum(np.exp(x - a)))

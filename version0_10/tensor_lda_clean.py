import numpy as np
import scipy
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot

# Import utility functions from other files
import tensor_lda_util as tl_util
import test_util

tl.set_backend('numpy')
device = 'cpu'#cuda

def get_M1(x):
    
    '''Get M1 moment by averaging all document vectors ((1) from [1])

    Parameters
    ----------
    x : ndarray (shape n_docs,n_words)

    Returns
    -------
    M1 : tensor of shape (1, x.shape[1]) (1,n_words)

    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    return tl.mean(x, axis = 0)

def get_M2(x, M1, alpha_0):
    '''Get M2 moment using (2) from [1]

    Parameters
    ----------
    x : ndarray

    M1 : tensor of shape (1, x.shape[1]) (1,n_words) equal to M1 moment (1) from [1]

    alpha_0 :  float equal to alpha_0 from [1] 

    Returns
    -------
    M2 : tensor of shape (x.shape[1], x.shape[1]) (n_words,n_words) equal to (2) from [1] (Co-occerence Matrix)

    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    sum_ = batched_tensor_dot(x, x) #(tensor outer product : produce (n_docs, n_words,n_words))
    sum_ = tl.mean(sum_, axis=0) #(n_words, n_words)
    sum_ = sum_ - np.diag(tl.mean(x, axis=0))
    sum_ *= (alpha_0 + 1)
    sum_ = sum_ - alpha_0*tl.reshape(kronecker([M1, M1]), sum_.shape) #intended shape is (n_words,n_words) cross means
    return sum_ 

def get_M3 (x, M1, alpha_0):
    '''Get M3 moment using (3) from [1]

    Parameters
    ----------
    x : ndarray (n_docs,n_words)

    M1 : tensor of shape (1, x.shape[1]) equal to M1 moment (1) from [1]

    alpha_0 : float equal to alpha_0 from [1]

    Returns
    -------
    M3 : tensor of shape (x.shape[1], x.shape[1], x.shape[1]) equal to (3) from [1]

    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    ns = x.shape[0] # n_docs
    n  = x.shape[1] # n_words

    #print(sum_)
    # first cross-moment term
    sum_ = batched_tensor_dot(x ,batched_tensor_dot(x, x))
    sum_ = tl.sum(sum_, axis=0)

    #issue: no dense equivalent
    # 2nd and 4th cross-moment terms
    diag = np.zeros((ns, n, n)) 
    for i in range(ns):
        diag[i] = np.diag(x[i])
    sum_ -= tl.sum(batched_tensor_dot(diag, x), axis=0) 
    sum_ -= tl.sum(batched_tensor_dot(x, diag), axis=0)

    # Last cross-moment term
    eye_mat = tl.eye(n)
    for _ in range(2):
        eye_mat = batched_tensor_dot(eye_mat, tl.eye(n))
    eye_mat = tl.sum(eye_mat, axis=0)
    sum_ += 2*eye_mat*tl.sum(x, axis=0)

    #final symmetric term
    tot = tl.zeros((1, n * n * n))
    for i in range(n):
        for j in range(n):
            tot += tl.sum(x[:,i]*x[:,j])*kronecker([tl_util.get_ei(n, i), tl_util.get_ei(n, j), tl_util.get_ei(n, i)])
    sum_ -= tl.reshape(tot, (n, n, n))
    sum_ *= (alpha_0 + 1)*(alpha_0+2)/(2*ns)

    M1_mat = tl.tensor([M1,]*n)*tl.sum(x, axis=0)[:, None] #(Matrix of M1)
    eye1 = tl.eye(n) # identity matrix
    eye2 = batched_tensor_dot(eye1, eye1) #outer product of two identity matrices
    tot2 = tl.sum(batched_tensor_dot(eye2, M1_mat), axis=0)+tl.sum(batched_tensor_dot(batched_tensor_dot(eye1, M1_mat), eye1), axis=0)+tl.sum(batched_tensor_dot(M1_mat, eye2), axis=0)
    sum_ -= alpha_0*(alpha_0 + 1)/(2*ns)*tot2 #rescale

    sum_ += alpha_0*alpha_0*(tl.reshape(kronecker([M1, M1, M1]), (n, n, n))) # get last term
    return sum_

def whiten(M, k, condition = False):
    '''Get W and W^(-1), where W is the whitening matrix for M, using the rank-k svd

    Parameters
    ----------
    M : tensor of shape (vocabulary_size, vocabulary_size) equal to
        the M2 moment tensor

    k : integer equal to the number of topics

    condition : bool, optional
        if True, print the M2 condition number

    Returns
    -------
    W : tensor of shape (vocabulary_size, number_topics) equal to the whitening
        tensor for M

    W_inv : tensor of shape (number_topics, vocabulary_size) equal to the inverse
        of W
    '''
    U, S, V = tl.partial_svd(M, n_eigenvecs=k) # right SV are n_words,n_topics
    #p = IncrementalPCA(k)
    #p.fit(M)
    #U = tl.transpose(p.components_)
    #S = p.explained_variance_*(M.shape[0] - 1)/(M.shape[0]) # pca.singular_values_
    W_inv = tl.dot(U, np.diag(tl.sqrt(S)))

    if condition == True:
        print("M2 condition number: " + str(np.amax(S)/np.amin(S)))

    return (U / tl.sqrt(S)[None, :]), W_inv
    
def topic_inference(term_counts, beta, gamma_shape, max_it, n_cols, alpha, display_not_converged = True):
    '''Infer the document-topic distribution vector for a given document

    Parameters
    ----------
    term_counts : array of length vocab_size equal to the number of occurrences
                  of each word in the vocabulary in a document

    beta : tensor of shape (number_topics, vocabulary_size) equal to the learned
           document-topic distribution

    gamma_shape : float equal to the shape parameter for the gamma distribution

    max_it : int equal to the maximum number of iterations before terminating

    n_cols : int equal to the number of weights (number of topics)

    alpha : array of length number_topics equal to the learned weights

    display_not_converged : bool, optional
        if True, print the iteration and change in any cases where the learned
        distribution does not converge

    Returns
    -------
    gammad : tensor of shape (1, n_cols) equal to the document/topic distribution
             for the term_counts vector
    '''
    gammad = tl.tensor(gamma.rvs(gamma_shape, scale= 1.0/gamma_shape, size = n_cols))
    exp_elogthetad = np.exp(tl_util.dirichlet_expectation(gammad))
    exp_elogbetad = np.array(beta)

    phinorm = np.add(np.dot(exp_elogbetad, exp_elogthetad), 1e-100)
    mean_gamma_change = 1.0
    cts_vector = term_counts

    iter = 0
    while (mean_gamma_change > 1e-3 and iter < max_it):
        lastgamma = tl.copy(gammad)
        gammad = (np.multiply(exp_elogthetad, (np.dot(exp_elogbetad.T, cts_vector / phinorm))) + alpha)
        exp_elogthetad = np.exp(tl_util.dirichlet_expectation(gammad))
        phinorm = np.add(np.dot(exp_elogbetad, exp_elogthetad), 1e-100)

        mean_gamma_change = np.sum(np.abs(gammad - lastgamma)) / n_cols
        all_gamma_change = gammad-lastgamma
        iter += 1
        if display_not_converged == True and iter % max_it == 0:
            print("iteration: " + str(iter) + ", change: " + str(mean_gamma_change))
            print("all change: " + str(all_gamma_change))

    return gammad

def non_negative_adjustment(M):
    '''Adjust M so that it is not negative by projecting it onto a simplex

    Parameters
    ----------
    M : tensor of shape (vocabulary_size, number_topics) equal to the learned
        factor

    Returns
    -------
    M_on_simplex : tensor of shape (vocabulary_size, number_topics) equal to the
        learned factor projected onto a simplex so it is non-negative
    '''
    M_on_simplex = np.zeros(M.shape)

    for i in range(0, M.shape[1]):
        projected_vector, theta = tl_util.simplex_proj(M[:, i] - np.amin(M[:, i]))
        projected_vector_revsign, theta_revsign = tl_util.simplex_proj(-1*M[:, i] - np.amin(-1*M[:, i]))

        if (theta < theta_revsign):
            M_on_simplex[:, i] = projected_vector
        else:
            M_on_simplex[:, i] = projected_vector_revsign
    return M_on_simplex

def fit(x, num_tops, alpha_0, verbose = True):
    '''Fit the documents to num_tops topics using the method of moments as
    outlined in [1]

    Parameters
    ----------
    x : ndarray of shape (number_documents, vocabulary_size) equal to the word
        counts in each document

    num_tops : int equal to the number of topics to fit x to

    alpha_0 : float equal to alpha_0 from [1]

    verbose : bool, optional
        if True, print the eigenvalues and best scores during the decomposition

    Returns
    -------
    w3_learned : tensor of shape (1, number_topics) equal to the weights
                 (eigenvalues) for each topic

    f3_reshaped : tensor of shape (number_topics, vocabulary_size) equal to the
                  learned topic/word distribution (requires adjustment using
                  the inference method)

    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    M1 = get_M1(x)
    M2_img = get_M2(x, M1, alpha_0)

    W, W_inv = whiten(M2_img, num_tops) # W (n_words x n_topics)
    X_whitened = tl.dot(x, W)     # this returns the whitened counts in  (n_topics x n_docs)
    M1_whitened = tl.dot(M1, W)   

    M3_final = get_M3(X_whitened, M1_whitened, alpha_0)
    # Because we use whitened M3, we need to unwhiten columnwise 
    # because sym_parafac returns (size,rank)
    lambdas_learned, phis_learned = sym_parafac(M3_final, rank=num_tops, n_repeat=15, verbose=verbose)

    # unwhiten
    phis_reshaped = (tl.dot(W_inv,phis_learned )) 
    return lambdas_learned, phis_reshaped 

def inference(x, factor, weights, gamma_shape = 1.0, n_its = 300, smoothing = 1e-6, display_not_conv = True):
    '''Infer the document/topic distribution from the factors and weights and
    make the factor non-negative

    Parameters
    ----------
    x : ndarray of shape (number_documents, vocabulary_size) equal to the word
        counts in each document

    factor : tensor of shape (number_topics, vocabulary_size) equal to the
             learned topic/word distribution

    weights : tensor of shape (1, number_topics) equal to the learned weights
              (eigenvalues) for each topic

    gamma_shape : float, optional
        the shape parameter for the gamma distribution

    n_its : int, optional
        the maximum number of iterations before exiting the inference method

    smoothing : float, optional
        the smoothing parameter for making factor nonzero

    display_not_conv : bool, optional
        if True print the iterations and change for document/topic distribution
        vectors that do not converge

    Returns
    -------
    gammad_norm2 : tensor of shape (number_documents, number_topics) equal to
                   the normalized document/topic distribution

    factor : tensor of shape (vocabulary_size, number_topics) equal to the
             adjusted factor
    '''
    adjusted_factor = factor
    #factor = non_negative_adjustment(factor)
    #factor = tl_util.smooth_beta(factor, smoothing=smoothing)
    adjusted_factor[adjusted_factor < 0.] = 0.
    # smooth beta
    adjusted_factor *= (1. - smoothing)
    adjusted_factor += (smoothing / adjusted_factor.shape[1])


    gammad_l = np.array([topic_inference(doc, factor, gamma_shape, n_its, len(weights), weights, display_not_converged = display_not_conv) for doc in x])
    gammad_l = np.nan_to_num(gammad_l)

    # normalize using exponential of dirichlet expectation
    gammad_norm = np.exp(np.array([tl_util.dirichlet_expectation(g) for g in gammad_l]))
    gammad_norm2 = np.array([row / np.sum(row) for row in gammad_norm])

    return gammad_norm2, adjusted_factor


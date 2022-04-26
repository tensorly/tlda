import math
import tensorly as tl
from   cumulant_gradient import cumulant_gradient
import tensor_lda_util as tl_util
from tensorly import check_random_state
import cupy as cp
import numpy as np
import pickle
import file_operations as fop
if(tl.get_backend() == "cupy"):
    from cupyx.scipy.special import gamma
else:
    from scipy.stats import gamma


def loss_rec(factor, cumulant, theta):
    '''Inputs:
        factor: (n_topics x n_topics): whitened factors from the SGD 
        cumulant: Whitened M3 (n_topics x n_topicsx n_topics)
        theta:  othogonalization penalty term (scalar)            
        output: 
        total loss evaluation: 
        orthogonality loss:
        reconstruction loss:   
    '''   

    rec = tl.cp_to_tensor((None, [factor]*3))
    rec_loss = 0
    if cumulant is not None:
        rec_loss = -1* tl.tenalg.inner(rec, cumulant)
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2 
    if cumulant is not None:
        return ortho_loss + rec_loss, ortho_loss, rec_loss/tl.norm(cumulant, 2)
    return ortho_loss + rec_loss, ortho_loss, rec_loss


class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, batch_size, learning_rate, cumulant = None, gamma_shape = 1.0, smoothing = 1e-6,theta=1,  ortho_loss_criterion = 1000,seed=None, dl = None): # we could try to find a more informative name for alpha_0
        
        if(tl.get_backend() == "cupy"):
            cp.random.seed(seed)
        else:
            np.random.seed(seed)

        self.n_topic = n_topic
        self.alpha_0 = alpha_0
        self.n_iter_train = n_iter_train
        self.n_iter_test  = n_iter_test
        self.batch_size   = batch_size
        self.learning_rate = learning_rate
        self.gamma_shape = gamma_shape
        self.smoothing   = smoothing
        self.theta       =  theta
        self.weights_    = tl.ones(self.n_topic)
        self.cumulant    = cumulant
        
        # Initial values 
        log_norm_std = 1e-5
        log_norm_mean = alpha_0
        ortho_loss = ortho_loss_criterion+1 # initializing the ortho loss
        i = 1
        # Finding optimal starting values based on orthonormal inits:
        while ortho_loss >= ortho_loss_criterion:
            if(tl.get_backend() == "cupy"):
                init_values = tl.tensor(cp.random.uniform(-1, 1, size=(n_topic, n_topic)))
            else:
                init_values = tl.tensor(np.random.uniform(-1, 1, size=(n_topic, n_topic)))
            init_values, _ = tl.qr(init_values, mode='reduced')
            _, ortho_loss, _ = tl_util.loss_rec(init_values, cumulant, self.theta)
            i += 1
            self.theta -= 0.1		
   
        self.factors_ = init_values
    def postprocess(self,pca,M1,vocab):
        '''Post-Processing '''
        # Postprocessing
        factors_unwhitened = pca.reverse_transform(self.factors_.T).T 

        #decenter the data
        factors_unwhitened += tl.reshape(M1,(vocab,1))
        factors_unwhitened [factors_unwhitened  < 0.] = 0. # remove non-negative probabilities
        
        # smooth beta
        factors_unwhitened *= (1. - self.smoothing)
        factors_unwhitened += (self.smoothing / factors_unwhitened.shape[1])
        factors_unwhitened /= factors_unwhitened.sum(axis=0)


        return factors_unwhitened
        
    def partial_fit(self, X_batch, learning_rate = None):
        '''Update the factors directly from the batch using stochastic gradient descent

        Parameters
        ----------
        X_batch : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors

        verbose : bool, optional
            if True, print information about every 200th iteration
        '''
        # incremental version
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.factors_ -= learning_rate*cumulant_gradient(self.factors_, X_batch, self.alpha_0,self.theta)

    def fit(self, X, pca=None, M1=None,vocab=None,verbose = True):
        '''Update the factors directly from X using stochastic gradient descent

        Parameters
        ----------
        X : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors
        '''
        tol = 1e-7 
        i   = 1
        max_diff = tol+1
        print("Fitting") 
        while (i <= 10 or max_diff >= tol) and i < self.n_iter_train:
            prev_fac = tl.copy(self.factors_)
            for j in range(0, len(X), self.batch_size):
                y  = X[j:j+self.batch_size]
                lr = self.learning_rate
                self.factors_ -= lr*cumulant_gradient(self.factors_, y, self.alpha_0, self.theta)
                self.factors_ /= tl.norm(self.factors_, axis=0)

            max_diff = tl.max(tl.abs(self.factors_ - prev_fac))
            i += 1
            if verbose and i%5 ==0:
                print(str(i)+"'th iteration complete. Maximum change in factors: "+str(max_diff))

        print("Total iterations: " + str(i))
        eig_vals = cp.array([np.linalg.norm(k)**3 for k in self.factors_ ])
        # normalize beta
        alpha           = cp.power(eig_vals, -2)
        alpha_norm      = (alpha / alpha.sum()) * self.alpha_0
        self.weights_   = tl.tensor(alpha_norm)
        
        # Convert whitened factors into word-topic probabilities
        if pca is not None and M1 is not None:
            self.factors_ = self.postprocess(pca,M1,vocab)


    def _predict_topic(self, X_batch):
        '''Infer the document-topic distribution vector for a given document

        Parameters
        ----------
        doc : tensor of length vocab_size equal to the number of occurrences
                      of each word in the vocabulary in a document

        adjusted_factor : tensor of shape (number_topics, vocabulary_size) equal to the learned
               document-topic distribution

        Returns
        -------
        gammad : tensor of shape (1, n_cols) equal to the document/topic distribution
                 for the doc vector
        '''

        # factors = nvocab x ntopics
        n_topics = len(self.weights_)
        n_docs = X_batch.shape[0]

        gammad = tl.tensor(gamma.rvs(self.gamma_shape, scale= 1.0/self.gamma_shape, size = (n_docs,n_topics)))
        exp_elogthetad = tl.tensor(cp.exp(tl_util.dirichlet_expectation(gammad))) #ndocs, n_topics
        phinorm = (tl.matmul(exp_elogthetad,self.factors_.T) + 1e-100) #ndoc X nvocab
        max_gamma_change = 1.0

        iter = 0
        while (max_gamma_change > 1e-3 and iter < self.n_iter_test):
            lastgamma      = tl.copy(gammad)
            gammad         = ((exp_elogthetad * (tl.matmul( X_batch / phinorm,self.factors_.T))) + self.weights_) # estimate for the variational mixing param
            exp_elogthetad = tl.exp(tl_util.dirichlet_expectation(gammad))
            phinorm        = (tl.matmul(exp_elogthetad,self.factors_.T) + 1e-20)

            mean_gamma_change_pdoc = tl.sum(tl.abs(gammad - lastgamma),axis=1) / n_topics
            max_gamma_change       = tl.max(mean_gamma_change_pdoc)
            iter += 1

        return gammad

    def predict(self, X_test):
        '''Infer the document/topic distribution from the factors and weights and
        make the factor non-negative

        Parameters
        ----------
        X_test : ndarray of shape (number_documents, vocabulary_size) equal to the word
            counts in each test document

        Returns
        -------
        gammad_norm2 : tensor of shape (number_documents, number_topics) equal to
                       the normalized document/topic distribution for X_test

        factor : tensor of shape (vocabulary_size, number_topics) equal to the
                 adjusted factor
        '''

        gammad_l = self._predict_topic(X_test)
        gammad_norm  = tl.exp([tl_util.dirichlet_expectation(g) for g in gammad_l])
        gammad_norm2 = gammad_norm/tl.reshape(tl.sum(gammad_norm,axis=1),(-1,1))

        return gammad_norm2

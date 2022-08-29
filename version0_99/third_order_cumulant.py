import tensorly as tl
from   .cumulant_gradient import cumulant_gradient
import numpy as np
# if(tl.get_backend() == "cupy"):
try:
    import cupy as cp
except ImportError:
    pass


def dirichlet_expectation(alpha):
    '''Normalize alpha using the dirichlet distribution'''
    return tl.digamma(alpha) - tl.digamma(sum(alpha))

def loss_rec(factor, theta):
    '''Inputs:
        factor: (n_topics x n_topics): whitened factors from the SGD 
        cumulant: Whitened M3 (n_topics x n_topicsx n_topics)
        theta:  othogonalization penalty term (scalar)            
        output:  
        orthogonality loss:
  
    '''   

    rec = tl.cp_to_tensor((None, [factor]*3))
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2 

    return ortho_loss 


class ThirdOrderCumulant():
    """Class to compute the third order cumulant

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, batch_size, 
                 learning_rate, gamma_shape=1.0,
                 theta=1, ortho_loss_criterion=1000, seed=None): # we could try to find a more informative name for alpha_0
        """"
        Computes the third order cumulant from centered, whitened batches of data, returns the learn factorized cumulant

        Parameters
        ----------
        n_topic : 
        alpha_0 : 
        n_iter_train : int
        n_iter_test : int
        batch_size : int
        learning_rate : float
        cumulant : 
        """
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
        # self.smoothing   = smoothing
        self.theta       =  theta
        # self.cumulant    = cumulant
        
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
            ortho_loss = tl_util.loss_rec(init_values, self.theta)
            i += 1
            self.theta -= 0.1		
   
        self.factors_ = init_values
    
        
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
        self.factors_ /= tl.norm(self.factors_, axis=0)

    def fit(self, X, verbose = True):
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
                self.partial_fit(y)

            max_diff = tl.max(tl.abs(self.factors_ - prev_fac))
            i += 1
            if verbose and i%5 ==0:
                print(str(i)+"'th iteration complete. Maximum change in factors: "+str(max_diff))

        print("Total iterations: " + str(i))


    def _predict_topic(self, X_batch, weights):
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
        n_topics = self.n_topic
        n_docs = X_batch.shape[0]

        gammad = tl.tensor(tl.gamma(self.gamma_shape, scale= 1.0/self.gamma_shape, size = (n_docs,n_topics)))
        exp_elogthetad = tl.tensor(tl.exp(tl_util.dirichlet_expectation(gammad))) #ndocs, n_topics
        phinorm = (tl.matmul(exp_elogthetad,self.factors_.T) + 1e-20) #ndoc X nvocab
        max_gamma_change = 1.0

        iter = 0
        while (max_gamma_change > 1e-3 and iter < self.n_iter_test):
            lastgamma      = tl.copy(gammad)
            gammad         = ((exp_elogthetad * (tl.matmul( X_batch / phinorm,self.factors_.T))) + weights) # estimate for the variational mixing param
            exp_elogthetad = tl.exp(tl_util.dirichlet_expectation(gammad))
            phinorm        = (tl.matmul(exp_elogthetad,self.factors_.T) + 1e-20)

            mean_gamma_change_pdoc = tl.sum(tl.abs(gammad - lastgamma),axis=1) / n_topics
            max_gamma_change       = tl.max(mean_gamma_change_pdoc)
            iter += 1

        return gammad

    def predict(self, X_test, weights):
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

        gammad_l = self._predict_topic(X_test, weights)
        gammad_norm  = tl.exp([tl_util.dirichlet_expectation(g) for g in gammad_l])
        gammad_norm2 = gammad_norm/tl.reshape(tl.sum(gammad_norm,axis=1),(-1,1))

        return gammad_norm2

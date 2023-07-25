import numpy as np
import tensorly as tl

from  .cumulant_gradient import cumulant_gradient


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
    """
    Class to compute the third order cumulant
    """
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, batch_size, 
                 learning_rate, gamma_shape=1.0,
                 theta=1, ortho_loss_criterion=1000, seed=None, n_eigenvec=None,
                 learning_rate_criterion = 1e-5): # we could try to find a more informative name for alpha_0
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
        rng = tl.check_random_state(seed)

        self.n_topic = n_topic
        self.alpha_0 = alpha_0
        self.n_iter_train = n_iter_train
        self.n_iter_test  = n_iter_test
        self.batch_size   = batch_size
        self.learning_rate = learning_rate
        self.gamma_shape = gamma_shape
        self.theta       =  theta
        if n_eigenvec is None:
            n_eigenvec = self.n_topic
        self.n_eigenvec = n_eigenvec
        self.learning_rate_criterion = learning_rate_criterion
        
        # initializing the orthogonality loss
        ortho_loss = ortho_loss_criterion+1

        # Finding optimal starting values based on orthonormal inits:
        while ortho_loss >= ortho_loss_criterion:
            init_values = tl.tensor(rng.uniform(-1, 1, size=(n_eigenvec, n_topic)))
                
            # init_values has shape (n_eigenvec, min(n_topic, n_eigenvec)) = (n_eigenvec, n_topic)
            init_values, _ = tl.qr(init_values, mode='reduced')
            ortho_loss = loss_rec(init_values, self.theta)

            self.theta -= 0.01
   
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
        del X_batch

    def fit(self, X, verbose = True):
        '''Update the factors directly from X using stochastic gradient descent

        Parameters
        ----------
        X : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors
        '''
        tol = self.learning_rate_criterion 
        i   = 1
        max_diff = tol+1
        
        while (i <= 10 or max_diff >= tol) and i < self.n_iter_train:
            prev_fac = tl.copy(self.factors_)
            for j in range(0, len(X), self.batch_size):
                y  = X[j:j+self.batch_size]
                self.partial_fit(y)
                del y

            max_diff = tl.max(tl.abs(self.factors_ - prev_fac))
            i += 1
            if verbose and i%5 ==0:
                print(str(i)+"'th iteration complete. Maximum change in factors: "+str(max_diff))
                
        del X
        print("Total iterations: " + str(i))


    def predict(self, X_test, adjusted_factors, weights):
        '''Infer the document/topic distribution from the factors and weights and
        make the factor non-negative

        Parameters
        ----------
        X_test : ndarray of shape (number_documents, vocabulary_size) equal to the word
            counts in each test document

        Returns
        -------
        gammad : tensor of shape (number_documents, number_topics) equal to
                       the normalized document/topic distribution for X_test
        '''

        # factors = nvocab x ntopics
        n_topics = self.n_topic
        n_docs = X_test.shape[0]

        gammad = tl.gamma(self.gamma_shape, scale= 1.0/self.gamma_shape, size = (n_docs,n_topics)) 
        exp_elogthetad = tl.exp(dirichlet_expectation(gammad)) #ndocs, n_topic
        
        epsilon = tl.finfo(gammad.dtype).eps
        phinorm = (tl.matmul(exp_elogthetad,adjusted_factors.T) + epsilon) #ndoc X nwords
        max_gamma_change = 1.0

        i = 0
        print("Begin Document Topic Prediction")
        while (max_gamma_change > 5e-3 and i < self.n_iter_test):
            lastgamma      = tl.copy(gammad)
            x_phi_norm     =  X_test / phinorm
            x_phi_norm_factors = tl.matmul(x_phi_norm, adjusted_factors)
            gammad         = ((exp_elogthetad * (x_phi_norm_factors)) + weights) # estimate for the variational mixing param
            exp_elogthetad = tl.exp(dirichlet_expectation(gammad)) 
            phinorm        = (tl.matmul(exp_elogthetad,adjusted_factors.T) + epsilon)

            mean_gamma_change_pdoc = tl.sum(tl.abs(gammad - lastgamma),axis=1) / n_topics
            max_gamma_change       = tl.max(mean_gamma_change_pdoc)
            i += 1
            print("End Document Topic Prediction Iteration " + str(i) +" out of "+str(self.n_iter_test))
            print("Current Maximal Change:" + str(max_gamma_change))
            
        del X_test
        return gammad

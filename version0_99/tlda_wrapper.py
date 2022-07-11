import tensorly as tl
from version0_99.second_order_cumulant import SecondOrderCumulant
from version0_99.third_order_cumulant import ThirdOrderCumulant



class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, learning_rate, pca_batch_size=10000, third_order_cumulant_batch=1000 , cumulant = None, gamma_shape = 1.0, smoothing = 1e-6, theta=1,  ortho_loss_criterion = 1000,seed=None):
        
        self.n_topic   = n_topic
        self.alpha_0   = alpha_0
        self.weights_  = tl.ones(self.n_topic)
        self.vocab = 0


        self.second_order = SecondOrderCumulant(n_topic, alpha_0, pca_batch_size)
        self.third_order = ThirdOrderCumulant(n_topic, alpha_0, n_iter_train, n_iter_test, third_order_cumulant_batch, learning_rate, cumulant, gamma_shape, smoothing, theta, ortho_loss_criterion, seed)

    def fit(self,X):
        self.vocab = X.shape[1]
        X_cent = X - tl.mean(X, axis=0)
        self.second_order.fit(X_cent)
        
        X_whit = self.second_order.transform(X_cent)
        self.third_order.fit(X_whit,verbose=False)

    def partial_fit(self, X_batch):
        # partial fit the mean and 2nd and 3rd order cumulants on the batch
        

        

    def postprocess(self):
        """
        uncenters and unwhitens the (factorized) third order cumulant 
        """
        # Postprocessing
        factors_unwhitened = self.second_order.reverse_transform(self.factors_.T).T 

        #decenter the data
        factors_unwhitened += tl.reshape(M1,(self.vocab,1))
        factors_unwhitened [factors_unwhitened  < 0.] = 0. # remove non-negative probabilities
        
        # smooth beta
        factors_unwhitened *= (1. - self.smoothing)
        factors_unwhitened += (self.smoothing / factors_unwhitened.shape[1])
        factors_unwhitened /= factors_unwhitened.sum(axis=0)


        return factors_unwhitened

    

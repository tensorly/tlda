import tensorly as tl
from version0_99.second_order_cumulant import SecondOrderCumulant
from version0_99.third_order_cumulant import ThirdOrderCumulant



class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, learning_rate, pca_batch_size=10000, third_order_cumulant_batch=1000 , cumulant = None, gamma_shape = 1.0, smoothing = 1e-6, theta=1,  ortho_loss_criterion = 1000,seed=None):
        
        self.n_topic   = n_topic
        self.alpha_0   = alpha_0
        self.smoothing = smoothing

        self.weights_  = tl.ones(self.n_topic)
        self.vocab = 0
        self.n_documents = 0
        self.mean = None

        self.second_order = SecondOrderCumulant(n_topic, alpha_0, pca_batch_size)
        self.third_order  = ThirdOrderCumulant(n_topic, alpha_0, n_iter_train, n_iter_test, third_order_cumulant_batch, learning_rate, cumulant, gamma_shape, theta, ortho_loss_criterion, seed)

    def fit(self,X):
        self.n_documents = X.shape[0]
        self.vocab = X.shape[1]
        self.mean = tl.mean(X, axis=0)

        X_cent = X - self.mean
        self.second_order.fit(X_cent)
        
        X_whit = self.second_order.transform(X_cent)
        self.third_order.fit(X_whit,verbose=False)

    def partial_fit(self, X_batch, iteration = 0):
        # partial fit the mean and 2nd and 3rd order cumulants on the batch
        if iteration == 0:
            if self.mean is None:
                self.vocab = X_batch.shape[1]
                self.mean = tl.mean(X_batch, axis=0)
            else:
                self.mean = ((self.mean * self.n_documents) + tl.sum(X_batch, axis=0)) / (self.n_documents + X_batch.shape[0])
            self.n_documents += X_batch.shape[0]
        elif iteration == 1:
            self.second_order.partial_fit(X_batch - self.mean)
        else:
            X_whit = self.second_order.transform(X_batch - self.mean)
            for j in range(0, len(X_whit), self.third_order_cumulant_batch):
                y  = X_whit[j:j+self.third_order_cumulant_batch]
                self.third_order.partial_fit(y, verbose=False)
    
    def transform(self, X, predict = True):
        # Postprocessing
        factors_unwhitened = self.second_order.reverse_transform(self.third_order.factors_.T).T 

        #decenter the data
        factors_unwhitened += tl.reshape(self.mean,(self.vocab,1))
        factors_unwhitened [factors_unwhitened  < 0.] = 0. # remove non-negative probabilities
        
        # smooth beta
        factors_unwhitened *= (1. - self.smoothing)
        factors_unwhitened += (self.smoothing / factors_unwhitened.shape[1])

        eig_vals = tl.tensor([tl.norm(k)**3 for k in self.third_order.factors_ ])
        alpha           = eig_vals**(-2)
        alpha_norm      = (alpha / alpha.sum()) * self.alpha_0
        self.weights_   = tl.tensor(alpha_norm)

        factors_unwhitened /= factors_unwhitened.sum(axis=0)

        if predict:
            predicted_topics = self.third_order.predict(X, self.weights_)
            return factors_unwhitened, predicted_topics
        
        return factors_unwhitened

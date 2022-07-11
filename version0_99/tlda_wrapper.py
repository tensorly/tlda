import tensorly as tl
from version0_99.second_order_cumulant import SecondOrderCumulant
from version0_99.third_order_cumulant import ThirdOrderCumulant



class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, learning_rate, pca_batch_size=10000, third_order_cumulant_batch=1000 , cumulant = None, gamma_shape = 1.0, smoothing = 1e-6, theta=1,  ortho_loss_criterion = 1000,seed=None):
        
        self.n_topic   = n_topic
        self.alpha_0   = alpha_0
        self.weights_  = tl.ones(self.n_topic)


        self.second_order = SecondOrderCumulant(n_topic, alpha_0, pca_batch_size)
        self.third_order = ThirdOrderCumulant(n_topic, alpha_0, n_iter_train, n_iter_test, third_order_cumulant_batch, learning_rate, cumulant, gamma_shape, smoothing, theta, ortho_loss_criterion, seed)

    def fit(self,X):
        X_cent = X - tl.mean(X, axis=0)
        self.second_order.fit(X_cent)
        
        
           


class TLDAOnline():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, learning_rate, pca_batch_size=10000, third_order_cumulant_batch=1000 , cumulant = None, gamma_shape = 1.0, smoothing = 1e-6, theta=1,  ortho_loss_criterion = 1000,seed=None):
        
        self.n_topic = n_topic
        self.alpha_0 = alpha_0
        self.n_iter_train  = n_iter_train
        self.n_iter_test   = n_iter_test
        self.learning_rate = learning_rate
        self.pca_batch_size   = pca_batch_size
        self.third_order_cumulant_batch = third_order_cumulant_batch
        self.gamma_shape = gamma_shape
        self.smoothing   = smoothing
        self.theta       =  theta
        self.weights_    = tl.ones(self.n_topic)
        self.cumulant    = cumulant
        self.ortho_loss_criterion = ortho_loss_criterion
        self.seed = seed


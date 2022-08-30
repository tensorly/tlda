import pickle
from pathlib import Path
import tensorly as tl

from .second_order_cumulant import SecondOrderCumulant
from .third_order_cumulant import ThirdOrderCumulant

class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, learning_rate, 
                pca_batch_size=10000, third_order_cumulant_batch=1000 , gamma_shape=1.0, smoothing=1e-6, 
                theta=1, ortho_loss_criterion=1000, random_seed=None):
        """
        n_topic : 
        alpha : 
        n_iter_train : int
        n_iter_test : int
        learning_rate : float
        pca_batch_size : int, default is 10K
        third_order_cumulant_batch = 1K
        
        """
        self.n_topic   = n_topic
        self.alpha_0   = alpha_0
        self.smoothing = smoothing
        self.third_order_cumulant_batch = third_order_cumulant_batch
        
        self.weights_  = tl.ones(self.n_topic)
        self.vocab = 0
        self.n_documents = 0
        self.mean = None
        self.unwhitened_factors_ = None

        self.second_order = SecondOrderCumulant(n_topic, alpha_0, pca_batch_size)
        self.third_order  = ThirdOrderCumulant(n_topic, alpha_0, n_iter_train, n_iter_test, third_order_cumulant_batch,
                                               learning_rate, gamma_shape, theta, ortho_loss_criterion, random_seed)

    def fit(self,X):
        """
        Compute the word-topic distribution for the entire dataset at once. Assumes that the whole dataset and 
        the tensors required to compute its word-topic distribution fit in memory.

        Parameters
        ----------
        X: tensor of size (self.n_documents , self.vocab) all documents used to fit the word-topic distribution
        """
        self.n_documents = X.shape[0]
        self.vocab = X.shape[1]
        self.mean = tl.mean(X, axis=0)

        X_cent = X - self.mean
        self.second_order.fit(X_cent)
        
        X_whit = self.second_order.transform(X_cent)
        self.third_order.fit(X_whit,verbose=False)

    
    def _partial_fit_first_order(self, X_batch):
        if self.mean is None:
            self.vocab = X_batch.shape[1]
            self.mean = tl.mean(X_batch, axis=0)
        else:
            self.mean = ((self.mean * self.n_documents) + tl.sum(X_batch, axis=0)) / (self.n_documents + X_batch.shape[0])
        self.n_documents += X_batch.shape[0]

    def _partial_fit_second_order(self, X_batch):
        self.second_order.partial_fit(X_batch - self.mean)
    
    def _partial_fit_third_order(self, X_batch):
        for j in range(0, len(X_batch), self.third_order_cumulant_batch):
            y  = X_batch[j:j+self.third_order_cumulant_batch]
            self.third_order.partial_fit(y) 

    def partial_fit(self, X_batch, batch_index, save_folder=None):
        """
        Update the word-topic distribution using a batch of documents. For a given batch, the
        first and second order cumulants need to be fit once, but the third order cumulant should
        be fit many times.
        
        Parameters
        ----------
        X_batch : tensor of shape (batch_size, self.vocab)
        batch_index : int
            index of the current batch.
            This is used to know whether to update the first and second moment or just whiten
        save_folder : str, default is None
            Folder in which to store the whitened batches.
            If None, the whitened batches will be recomputed at each iteration
            instead of being catched.
        """
        if not hasattr(self, "seen_batches"):
            self.seen_batches = dict()
        
        if batch_index in self.seen_batches:
            # We've seen the batch at least once
            if self.seen_batches[batch_index] != 0:
                # We already whitened it, just load that
                if save_folder:
                    save_file = self.seen_batches[batch_index]
                    X_batch = pickle.load(open(Path(save_folder).joinpath(save_file).as_posix(),'rb'))
                else:
                    X_batch = self.second_order.transform(X_batch - self.mean)

            else:
                # We only saw it once: that whitened version is not exact, recompute
                X_batch = self.second_order.transform(X_batch - self.mean)
                if save_folder is not None:
                    save_file = f'_whitened_batch_{batch_index}'
                    self.seen_batches[batch_index] = save_file
                    pickle.dump(X_batch, open(Path(save_folder).joinpath(save_file).as_posix(), 'wb'))
                else:
                    self.seen_batches[batch_index] = 1
            
            self._partial_fit_third_order(X_batch)

        else:
            # First time we see the batch: recompute the whitened version next time
            self._partial_fit_first_order(X_batch)
            self._partial_fit_second_order(X_batch)
            self.seen_batches[batch_index] = 0


    def partial_fit_online(self, X_batch):
        """
        Update the word-topic distribution using a batch of documents in a fully online version. Meant for very large datasets,
        since we only do one gradient update for each batch in the third order cumulant calculation.
        
        Parameters
        ----------
        X_batch : tensor of shape (batch_size, self.vocab)
        """        
        self._partial_fit_first_order(X_batch)
        self._partial_fit_second_order(X_batch)
        X_batch = self.second_order.transform(X_batch - self.mean)
        self._partial_fit_third_order(X_batch)

    def _unwhiten_factors(self):
        """Unwhitens self.third_order.factors_, then uncenters and unnormalizes"""
        factors_unwhitened = self.second_order.reverse_transform(self.third_order.factors_.T).T 

        # Un-centers the data
        factors_unwhitened += tl.reshape(self.mean,(self.vocab,1))
        factors_unwhitened [factors_unwhitened  < 0.] = 0. # remove non-negative probabilities
        
        # Smoothing
        factors_unwhitened *= (1. - self.smoothing)
        factors_unwhitened += (self.smoothing / factors_unwhitened.shape[1])

        # Calculate the eigenvalues from the whitened factors
        eig_vals = tl.tensor([tl.norm(k)**3 for k in self.third_order.factors_ ])
        alpha           = eig_vals**(-2)
        # Recover the topic weights 
        alpha_norm      = (alpha / alpha.sum()) * self.alpha_0
        self.weights_   = tl.tensor(alpha_norm)

        # Normalize the factors
        factors_unwhitened /= factors_unwhitened.sum(axis=0)
        return factors_unwhitened

    @property
    def unwhitened_factors(self):
        """Unwhitened learned factors of shape (n_topic, vocabulary_size)

        On the first call, this will compute and store the unwhitened factors.
        Subsequent calls will simply return the stored value. 
        """
        if self.unwhitened_factors_ is None:
            self.unwhitened_factors_ = self._unwhiten_factors()
        else:
            return self.unwhitened_factors_

    def transform(self, X=None, predict=True):
        """
        Transform the document-word matrix of a set of documents into a word-topic distribution and topic-distribution when predict=True.

        Parameters
        ----------  
        X : tensor of shape (n_documents , self.vocab) 
            set of documetns to predict topic distribution
        predict : indicate whether to return topic-document distribution and word-topic distribution or just word-topic distribution. 
        """
        print(self.unwhitened_factors_)
        self.third_order.unwhitened_factors_ = self.unwhitened_factors_
        if predict:
            predicted_topics = self.third_order.predict(X, self.weights_)
            return predicted_topics
        
        return predicted_topics

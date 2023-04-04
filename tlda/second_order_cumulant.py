import tensorly as tl
from sklearn.decomposition import IncrementalPCA
try:
    import cuml
except ImportError:
    pass

class SecondOrderCumulant():
    """Class to compute the second order cumulant

    Parameters
    ----------
    n_eigenvec : int
        Corresponds to the number of topics in the Tensor LDA
    alpha_0 : int
        Mixing parameter for the topic weights
    batch_size : int
        Size of the batch to use for online learning
    n_docs : int
        Running count of fitted documents. Used for normalization
    """
    def __init__(self, n_eigenvec, alpha_0, batch_size): # n_eigenvec here corresponds to n_topic in the LDA
        self.n_eigenvec = n_eigenvec
        self.alpha_0 = alpha_0
        self.batch_size = batch_size
        self.n_docs = 0
        if tl.get_backend() == "numpy":
            self.pca = IncrementalPCA(n_components = self.n_eigenvec, batch_size = self.batch_size)
        elif tl.get_backend()  == "cupy":
            self.pca = cuml.IncrementalPCA(n_components = self.n_eigenvec, batch_size = self.batch_size)


    def fit(self, X):
        '''Fit the entire data to get the projection weights (singular vectors) and
        whitening weights (scaled explained variance) of a centered input dataset X.

        Parameters
        ----------
        X : tensor of shape (n_samples, vocabulary_size)
            Tensor containing all input documents
        '''
        self.n_docs += X.shape[0]
        
        self.pca.fit(X*tl.sqrt(self.alpha_0+1))
        self.projection_weights_ = tl.transpose(self.pca.components_)
        self.whitening_weights_ = self.pca.explained_variance_*(self.n_docs - 1)/(self.n_docs)
        del X

    def partial_fit(self, X_batch):
        '''Fit a batch of data and update the projection weights (singular vectors) and
        whitening weights (scaled explained variance) accordingly using a centered
        batch of the input dataset X.

        Parameters
        ----------
        X_batch : tensor of shape (batch_size, vocabulary_size)
            Tensor containing a batch of input documents
        '''
        self.n_docs += X_batch.shape[0]
        
        self.pca.partial_fit(X_batch*tl.sqrt(self.alpha_0+1))
        self.projection_weights_ = tl.transpose(self.pca.components_)
        self.whitening_weights_ = self.pca.explained_variance_*(self.n_docs - 1)/(self.n_docs)
        del X_batch

    def transform(self, X):
        '''Whiten some centered tensor X using the fitted PCA model.

        Parameters
        ----------
        X : tensor of shape (batch_size, vocabulary_size)
            Batch of centered samples

        Returns
        -------
        whitened_X : tensor of shape (batch_size, self.n_eigenvec)
            Whitened samples 
        '''
        X_whit = tl.dot(X, (self.projection_weights_ / tl.sqrt(self.whitening_weights_)[None, :]))
        del X
        return X_whit

    def reverse_transform(self, X):
        '''Unwhiten some whitened tensor X using the fitted PCA model.

        Parameters
        ----------
        X : tensor of shape (batch_size, self.n_eigenvec)
            whitened input tensor

        Returns
        -------
        unwhitened_X : tensor of shape (batch_size, vocabulary_size)
            Batch of unwhitened centered samples
        '''
        X_unwhit = tl.dot(X, (self.projection_weights_ * tl.sqrt(self.whitening_weights_)).T)
        del X
        return X_unwhit
        #return tl.dot(X, (self.projection_weights_ / tl.sqrt(self.whitening_weights_)[None, :]).T)

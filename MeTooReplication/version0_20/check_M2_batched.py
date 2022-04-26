import tensorly as tl
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_almost_equal 
import pickle
import cupy as cp
import os
backend ="cupy"
tl.set_backend(backend)
device  = 'cuda'#cuda

# Load Data
pca      = pickle.load(open('../data/pca.obj','rb'))
M1       = pickle.load(open('../data/M1.obj','rb'))

def get_batched_M2(x, alpha_0):
    return (alpha_0 + 1)*tl.sum(batched_tensor_dot(x, x), axis=0)




def get_M2(x, alpha_0):
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
    sum_ *= (alpha_0 + 1)
    return sum_ 

alpha_0  = 0.01
k        = 15
true_res = tl.eye(k)

inDir  = "../data/MeTooMonthCleaned/" # input('Name of input directory? : ')
dl     = os.listdir(inDir)


## Calculate Batched M2
batch_size = 500
M2 = None
tot_len = 0



for f in dl:
    a_cent = pickle.load( open('../data/x_mat/' + f[:-4] + '.obj','rb'))
    a_cent -=  M1
    tot_len += a_cent.shape[0]
    print("Length: "+str(tot_len))
    print("Begin: "+f)    
    for j in range(0, len(a_cent)-(batch_size-1), batch_size):
        a_batch = a_cent[j:j+batch_size]
        if M2 is None:
            M2 = get_batched_M2(a_batch, alpha_0)
        else:
            M2 += get_batched_M2(a_batch, alpha_0)
        if j % 10000==0:
            print("Completed: "+str(j)+"/" +str(len(a_cent)) )
    del a_cent

M2 /= tot_len


W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
res = tl.dot(tl.dot(W.T, M2), W)
assert_array_almost_equal(res, true_res)
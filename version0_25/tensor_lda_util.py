import numpy as np
import cupy as cp
import tensorly as tl
if(tl.get_backend() == "cuda"):
	from cupyx.scipy.special import digamma, gammaln
else:
	from scipy.special import  digamma, gammaln

# import sparse


# Import TensorLy
import tensorly as tl
from tensorly import norm
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot




def dirichlet_expectation(alpha):
    '''Normalize alpha using the dirichlet distribution'''
    return digamma(alpha) - digamma(sum(alpha))





def log_sum_exp(x):
    '''calculate log(sum(exp(x)))'''
    a = tl.max(x)
    
    if(tl.get_backend() == "cuda"):
        return a + cp.log(cp.sum(cp.exp(x - a)))
    else:
        return a + np.log(np.sum(np.exp(x - a)))

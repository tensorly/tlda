import tensorly as tl
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner
from tensorly import check_random_state
import cupy as cp
import numpy as np

def init_factor(n_topic,seed=None):
    # None seed uses numpy global seed
    # r = check_random_state(seed)
    std = 1e-5
    order = 3 # always looking for the 3rd order moment
    #std_factors = (std/tl.sqrt(n_topic))**(1/order)
    # ensure initial values are on proper scale    
    if (seed is not None):
        if tl.get_backend() == "cupy":
            cp.random.seed(seed)
        else:
            np.random.seed(seed)

    if (tl.get_backend() == "cupy"):
        init_values = tl.tensor(cp.random.normal(-1, 1, size=(n_topic, n_topic)))
    else:
        init_values = tl.tensor(np.random.normal(-1, 1, size=(n_topic, n_topic)))
    init_values, _ = tl.qr(init_values, mode='reduced')
    #tl.abs(tl.tensor(cp.random.normal(0, std_factors, size=(n_topic, n_topic))))
    print("Initialization")
    print(init_values)

    return init_values

def cumulant_gradient(phi, y, y_mean, alpha, theta=1):
    rank = phi.shape[1]

    grad_weight = tl.zeros(phi.shape)
    for i in range(rank):
        for j in range(rank):
            grad_weight[:, i] -= (1 + theta)/2*phi[:, j]*inner(phi[:, j], phi[:, i])**2
        grad_weight[:, i] +=  (1 + alpha)*(2 + alpha)/2*y*inner(phi[:, i], y)*inner(phi[:, i], y)
        grad_weight[:, i] += (alpha**2)*y_mean*inner(phi[:, i], y_mean)*inner(phi[:, i], y_mean)
        grad_weight[:, i] -= alpha*(1 + alpha)/2*y_mean*inner(phi[:, i], y)*inner(phi[:, i], y)
        grad_weight[:, i] -= alpha*(1 + alpha)/2*y*inner(phi[:, i], y)*inner(phi[:, i], y_mean)
        grad_weight[:, i] -= alpha*(1 + alpha)/2*y*inner(phi[:, i], y_mean)*inner(phi[:, i], y)

        
    return -1*grad_weight

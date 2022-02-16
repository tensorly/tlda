import tensorly as tl
# This should work with any backend
# tl.set_backend('pytorch')
from tensorly.cp_tensor import cp_mode_dot
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner

def cumulant_gradient(phi, y, alpha, theta=10):
    """Returns the gradient of the third order cumulant with respect to the factors Phi

    Parameters
    ----------
    phi : tensor of shape (n_features, rank)
        the factors we are trying to learn
        We are decomposing the actual cumulant tensor T as phi o phi o phi (where o is outer product)
    y : the centered parameter of the actual cumulant
    alpha :
    theta : int, default is 1
    """

    #print(tl.get_backend())
    gradient = 3*(1 + theta)*tl.dot(phi, tl.dot(phi.T, phi)**2)
    # gradient = 2*(1 + theta)*(tl.dot(phi, (tl.dot(phi.T, phi) - tl.eye(phi.shape[0]))) + tl.dot(phi.T, (tl.dot(phi, phi.T) - tl.eye(phi.shape[0]))))
    # gradient = 2*(1 + theta)*(tl.dot(phi, (tl.dot(phi.T, phi) - tl.eye(phi.shape[0]))))
    # gradient = 3*(1 + theta)*tl.dot(phi, (tl.dot(phi.T, phi)**2 - tl.eye(phi.shape[0])*tl.diag(phi)))
    # gradient -= 3*(1 + alpha)*(2 + alpha)/(2*y.shape[0])*tl.dot(y.T, tl.dot(y, phi)**2)
    gradient -= 3*(1 + alpha)*(2 + alpha)/(2*y.shape[0])*tl.dot(y.T, tl.dot(y, phi)**2)
    return gradient




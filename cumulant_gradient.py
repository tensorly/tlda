import tensorly as tl

def cumulant_gradient(phi, y, alpha, theta=10):
    """Returns the gradient of loss with respect to the factors Phi

    The loss corresponds to the reconstruction error of the third order cumulant (squared Frobenius norm)
    plus an orthogonality term on the learned factors (equation 11 in the paper).

    Parameters
    ----------
    phi : tensor of shape (n_features, rank)
        the factors we are trying to learn
        We are decomposing the actual cumulant tensor T as phi o phi o phi (where o is outer product)
    y : the centered parameter of the actual cumulant
    alpha : 
    theta : int, default is 1

    Returns
    -------
    gradient : tensor of shape (n_features, rank)
        d(loss)/d(Phi)
    """
    gradient = 3*(1 + theta)*tl.dot(phi, tl.dot(phi.T, phi)**2)
    gradient -= 3*(1 + alpha)*(2 + alpha)/(2*y.shape[0])*tl.dot(y.T, tl.dot(y, phi)**2)
    return gradient



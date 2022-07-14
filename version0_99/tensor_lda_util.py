import tensorly as tl


def dirichlet_expectation(alpha):
    '''Normalize alpha using the dirichlet distribution'''
    return tl.digamma(alpha) - tl.digamma(sum(alpha))




def loss_rec(factor, theta):
    '''Inputs:
        factor: (n_topics x n_topics): whitened factors from the SGD 
        cumulant: Whitened M3 (n_topics x n_topicsx n_topics)
        theta:  othogonalization penalty term (scalar)            
        output:  
        orthogonality loss:
  
    '''   

    rec = tl.cp_to_tensor((None, [factor]*3))
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2 

    return ortho_loss 




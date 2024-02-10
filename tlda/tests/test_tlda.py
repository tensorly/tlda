from ..tlda_wrapper import TLDA
import tensorly as tl
import numpy as np
import numpy.random

def get_mu(top_n, vocab_size, doc_num, t_per_doc,seed):
    tl.set_backend('numpy')
    '''use code here:
    http://www.hongliangjie.com/2010/09/30/generate-synthetic-data-for-lda/
    to get document, topic matrices'''
    ## define some constant
    np.random.seed(seed=seed)
    TOPIC_N = top_n
    VOCABULARY_SIZE = vocab_size
    DOC_NUM = doc_num
    TERM_PER_DOC = t_per_doc
    w_arr = tl.zeros((DOC_NUM, VOCABULARY_SIZE), dtype=tl.float32)

    #beta = [0.01 for i in range(VOCABULARY_SIZE)]
    #alpha = [0.9 for i in range(TOPIC_N)]
    beta = [0.1 for i in range(VOCABULARY_SIZE)]
    alpha = [0.01 for i in range(TOPIC_N)]

    mu = []
    theta_arr = np.zeros((DOC_NUM, TOPIC_N))
    ## generate multinomial distribution over words for each topic
    for i in range(TOPIC_N):
        topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
        mu.append(topic)

    for i in range(DOC_NUM):
        buffer = {}
        z_buffer = {} ## keep track the true z
        ## first sample theta
        theta = numpy.random.mtrand.dirichlet(alpha,size = 1)
        # theta /= np.sum(theta)
        for j in range(TERM_PER_DOC):
            ## first sample z
            z = numpy.random.multinomial(1,theta[0],size = 1)
            z_assignment = 0
            for k in range(TOPIC_N):
                if z[0][k] == 1:
                    break
                z_assignment += 1
            if not z_assignment in z_buffer:
                z_buffer[z_assignment] = 0
            z_buffer[z_assignment] = z_buffer[z_assignment] + 1
    		## sample a word from topic z
            w = numpy.random.multinomial(1,mu[z_assignment][0],size = 1)
            w_assignment = 0
            for k in range(VOCABULARY_SIZE):
                if w[0][k] == 1:
                    break
                w_assignment += 1
            if not w_assignment in buffer:
                buffer[w_assignment] = 0
            buffer[w_assignment] = buffer[w_assignment] + 1
            w_arr[i] = w_arr[i] + w
        theta_arr[i] = theta
    return tl.tensor(w_arr), mu, theta_arr, sum(alpha)

def correlate(a, v):
    a = (a - tl.mean(a)) / (np.std(a) * len(a))
    v = (v - tl.mean(v)) /  np.std(v)
    return np.correlate(a, v)

def test_fit():
    # set parameters
    k = 2
    num_docs = 2000
    density = 15
    vocab   = 100
    n_iter_train     = 2001
    batch_size_pca =  2000
    batch_size_grad  = 10
    n_iter_test = 10
    learning_rate = 0.01
    seed       = 1

    # get dataset from dirichlet distribution
    x, mu, _, alpha_0 = get_mu(k, vocab, num_docs, density, seed)

    # run TLDA
    tlda = TLDA(
        k, alpha_0, n_iter_train, n_iter_test, learning_rate,
        pca_batch_size=batch_size_pca,
        third_order_cumulant_batch=batch_size_pca,
        random_seed=seed
    )
    tlda.fit(x)
    factors_unwhitened = tlda.unwhitened_factors

    # test correlation between predicted and ground truth topics
    mu = np.asarray(mu)[:, 0, :]
    permutations = [[0, 1], [1, 0]]

    accuracy = []
    for j in range(len(permutations)):
        for i in range(k):
            accuracy.append(correlate(factors_unwhitened.T[i,:], mu[permutations[j]][i,:]))

    assert(max(sum(accuracy[:2]), sum(accuracy[2:])) >= 1.5)

test_fit()

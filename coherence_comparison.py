import numpy as np
import cupy as cp
import scipy
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation as sklearn_LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal
# from cumulant_gradient import cumulant_gradient

import time
import csv

#Insert Plotly
import matplotlib.pyplot as plt
import pickle
# Import utility functions from other files
from version0_20.tlda_final import TLDA
from version0_20.pca import PCA
import version0_20.test_util as test_util
import version0_20.tensor_lda_util as tl_util
from version0_20.preprocess_efficient import cleanLine, regexchars, tokenize, removeStopwords, stem

import version0_15.tensor_lda_clean as tlda_mid

import gensim.corpora as corpora
import gensim.models as models

#backend="cupy"
backend = "numpy"
tl.set_backend(backend)

VOCAB = 1000

def coherence_mean_npmi (term_indices, tcm, smooth, n_doc_tcm):
    #given suitably ordered pairs of indices stored in two column matrix "indices" a non-vectorized calculation would be something like
    #mapply(function(x, y)  {(log2((tcm[x,y]/n_doc_tcm) + smooth) - log2(tcm[x,x]/n_doc_tcm) - log2(tcm[y,y]/n_doc_tcm)) / -log2((tcm[x,y]/n_doc_tcm) + smooth)}}    #                        , indices[,1], indices[,2])
    tl.set_backend('numpy')
    if n_doc_tcm <= 0:
        return 0
    res = None
    n = len(term_indices)
    if(n >= 2):
        res = tcm[np.ix_(term_indices, term_indices)] / n_doc_tcm
        res[np.triu_indices(n, k=1)] = res[np.triu_indices(n, k=1)] + smooth
        # interim storage of a denominator
        denominator =  res[np.triu_indices(n, k=1)]
        d = np.diag(res)
        res = (res.T/d).T
        res = np.dot(res, np.diag(1 / d))
        res = res[np.triu_indices(n, 1)]
        res = np.log2(res) / -np.log2(denominator)
        res = np.mean(res, where = ~np.isnan(res))
    return res

def coherence_mean_npmi_cosim(term_indices, tcm, smooth, n_doc_tcm):
    #TODO
    #example of nonvectorized calculation
    tl.set_backend('numpy')
    if n_doc_tcm <= 0:
        return 0
    res = None
    n = len(term_indices)
    if n >= 2:
        res = tcm[np.ix_(term_indices, term_indices)] / n_doc_tcm
        res[np.tril_indices(n, k = -1)] = (res.T)[np.tril_indices(n, k = -1)]
        res = res + smooth
        res[np.diag_indices(n)] = np.diag(res) - smooth
        #interim storage of denominator
        denominator =  res
        d = np.diag(res)
        res = (res.T/d).T
        res = np.dot(res, np.diag(1 / d))
        res = np.log2(res) / -np.log2(denominator)
        #create values for cosine similarity check, for this metric: the sum of all npmi values
        res_compare = (np.reshape(np.tile(np.ndarray.sum(res, axis=0), n), (n, n))).T
        res_norm = preprocessing.normalize(res, norm='l2')
        res_compare_norm = preprocessing.normalize(res_compare, norm='l2')
        res = metrics.pairwise.cosine_similarity(res_norm, res_compare_norm)
        res = np.mean(res, where = ~np.isnan(res))
    return(res)

def get_uci_data():
    tl.set_backend('numpy')
    test_len = None
    #test_len = 1000

    '''preprocess'''
    #categories = ['sci.med', 'rec.sport.baseball']#'rec.sport.baseball', 'soc.religion.christian', 'rec.sport.baseball', 'sci.space', 'rec.autos'
    newsgroups_test = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))#, categories=categories) #remove=('headers', 'footers', 'quotes') ,  remove=('headers', 'footers', 'quotes'), categories = categories
    if test_len is not None:
        texts = newsgroups_test.data[:test_len]
    else:
        texts = newsgroups_test.data
    print("text length: ")
    print(len(texts))
    texts = [stem(removeStopwords(tokenize(regexchars(cleanLine(line))))) for line in texts]
    # texts = lemmatize(texts)
    # texts = make_bigrams(texts)
    #lengths = np.array([len(text) for text in texts])
    # texts2 = np.array([' '.join(row) for row in texts])
    # print(texts)
    vectorizer = CountVectorizer(max_df = 0.1, min_df = 0.005)
    vectors = vectorizer.fit_transform(texts).toarray()
    vocab = vectorizer.get_feature_names()
    print(len(vocab))
    print(len(vectorizer.stop_words_))

    texts2 = []
    for text in texts:
        texts2.append([w for w in text.split(' ') if w.lower() in vectorizer.vocabulary_])

    return vectors, vocab, texts2

def postprocess(factors_unwhitened, x, vocab, num_tops, smoothing, decenter=False, name="", alpha_0 = 1):
    '''Post-Processing '''
    res = []
    # Postprocessing

    #This is hard-coded. We should calculate the alphas by hand. 
    if decenter == True:
        #eig_vals = cp.array([np.linalg.norm(k)**3 for k in factors_unwhitened.T ])
        # normalize beta
        #alpha           = cp.power(eig_vals, -2)
        #alpha_norm      = (alpha / alpha.sum()) * alpha_0
        #weights   = tl.tensor(alpha_norm)
        #print("weights shape:")
        #print(weights.shape)

        #fac2 = factors_unwhitened/weights
        #print("fac2 shape: ")
        #print(fac2.shape)
        #fac2 = (fac2.T + tl.mean(x, axis=0)).T
        #fac2 *= weights
        #fac2 = cp.asarray(fac2)
        #print("final fac2: ")
        #print(fac2)

        #print("decenter with new strategy:")
        #print(fac2[0])
        t1 = time.time()
        #wc   =  cp.asarray(tl.mean(x, axis=0))#/vocab*(1/num_tops)
        wc = np.asarray(tl.mean(x, axis=0))
        wc   =  tl.reshape(wc,(vocab,1))
        
        factors_unwhitened = np.asarray(factors_unwhitened)
        #factors_unwhitened   =  cp.asarray(factors_unwhitened)
        factors_unwhitened += wc
        t2 = time.time()
        print("Decentering: " + str(t2-t1))
        res.append((name + ' decentering', t2-t1))
        print("decenter with old strategy:")
        print(factors_unwhitened[0])

    #print(factors_unwhitened.dtype)
    #print(wc.dtype)
    #print(factors_unwhitened.shape)
    #print(wc.shape)

    #factors_unwhitened   =  cp.asarray(factors_unwhitened)
    factors_unwhitened = np.asarray(factors_unwhitened)
    # print(factors_unwhitened)
    t1 = time.time()
    factors_unwhitened [factors_unwhitened  < 0.] = 0.
    # smooth beta
    factors_unwhitened  *= (1. - smoothing)
    #print(factors_unwhitened)

    factors_unwhitened += (smoothing / factors_unwhitened.shape[1])
    #print(factors_unwhitened)
    #print("begin print estimated mu")
    factors_unwhitened /= factors_unwhitened.sum(axis=0)
    t2 = time.time()
    print("Smoothing and Normalization: " + str(t2-t1))
    res.append((name + ' smoothing and normalization', t2-t1))
    #print(factors_unwhitened)
    # remean the data
    #print("begin mean")
    #if decenter == True:
    #    fac2[fac2 < 0.] = 0.
    #    fac2 *= (1. - smoothing)
    #    fac2 += (smoothing/fac2.shape[1])
    #    fac2 /= fac2.sum(axis=0)
    # print(wc)
    # print("begin ground truth")
    # print(mu)


    """ INSERT CODE FOR COHERENCE HERE """
    # if decenter == True:
    #    return res, fac2
    return res, factors_unwhitened


def gen_fit_0_20(x, num_tops = 2, alpha_0 = 0.01, n_iter_train = 2001):
    vocab   = x.shape[1]
    n_iter_train     = n_iter_train
    batch_size_pca =  20000 # 2000
    batch_size_grad  = 10
    n_iter_test = 10 
    theta_param =  0.5
    learning_rate = 0.0001
    smoothing  = 1e-5 #1e-5

    # res = {}
    res = []

    #backend="cupy"
    backend = "numpy"
    tl.set_backend(backend)
    
    x = tl.tensor(x)


    t1 = time.time()
    x_cent = tl.tensor(x - tl.mean(x, axis=0))
    t2 = time.time()
    print("Centering time: " + str(t2-t1))
    res.append(('centering', t2-t1))

    t1 = time.time()
    pca = PCA(num_tops, alpha_0, batch_size_pca,backend)
    pca.fit(x_cent)
    t2 = time.time()
    print("PCA fit: " + str(t2-t1))
    res.append(('PCA fit', t2-t1))

    # M2_img = tlda_mid.get_M2(x_cent, tl.mean(x_cent, axis=0), alpha_0)
    # W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    # print(tl.dot(tl.dot(W.T, M2_img), W))


    t1 = time.time()
    # W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    x_whit = pca.transform(x_cent)
    t2 = time.time()
    print("PCA Transform: " + str(t2-t1))
    res.append(('PCA transform', t2-t1))

    '''fit the tensor lda model'''
    t1 = time.time()
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,gamma_shape = 1.0, smoothing = smoothing,theta=theta_param)
    tlda.fit(x_whit,verbose=False)
    t2 = time.time()
    print("TLDA fit: " + str(t2-t1))
    res.append(('TLDA fit', t2-t1))

    t1 = time.time()
    factors_unwhitened = pca.reverse_transform(tlda.factors_.T)
    factors_unwhitened = factors_unwhitened.T
    t2 = time.time()
    print("PCA Reverse Transform: " + str(t2-t1))
    res.append(('unwhiten factors', t2-t1))
    
    res2, factors_unwhitened = postprocess(factors_unwhitened, x, vocab, num_tops, smoothing, True, alpha_0 = alpha_0)
    #tl.mean(cp.asnumpy(accuracy))
    # {**res, **res2}
    res.extend(res2)
    return res, factors_unwhitened


def main():
    n_tops = 20
    alpha_0 = 0.1
    
    print("starting")
    texts, vocab, texts_lemmatized = get_uci_data()
    print("got data")
    print(texts.shape)

    tcm = tl.dot(texts.T, texts)

    t1 = time.time()
    lda = sklearn_LDA(n_components = n_tops, doc_topic_prior = alpha_0, max_iter = 10)
    lda.fit(texts)
    factors_sklearn = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    t2 = time.time()
    res2 = ('sklearn LDA', t2 - t1)
    print(res2)
    
    id2word = corpora.Dictionary(texts_lemmatized)
    corpus = [id2word.doc2bow(text) for text in texts_lemmatized]
    t1 = time.time()
    lda_model = models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=n_tops,
                                       passes=10,
                                       alpha=alpha_0,
                                       per_word_topics=False)
    factors_gensim = lda_model.get_topics()
    t2 = time.time()
    res3 = ('gensim LDA', t2 - t1)
    print(res3)

    res, factors_tlda = gen_fit_0_20(texts, num_tops = n_tops, alpha_0 = alpha_0, n_iter_train = 10001)
    factors_tlda = factors_tlda.T
    print(res)
    print(factors_tlda.shape)
    tl.set_backend('numpy')

    K = 20
    vocab = np.asarray(vocab)
    #texts = cp.asnumpy(texts)
    #texts_lemmatized = cp.asnumpy(texts_lemmatized)
    # tcm = tl.mean(batched_tensor_dot(texts, texts), axis=0)
    print(tcm.shape)
    n_doc_tcm = len(texts)
    smooth = 0.0001

    sklearn_coherence  = []
    gensim_coherence = []
    tlda_coherence = []
    sklearn_cosim = []
    gensim_cosim = []
    tlda_cosim = []

    print(factors_sklearn.shape)
    print(factors_gensim.shape)
    print(factors_tlda.shape)

    #factors_tlda = cp.asnumpy(factors_tlda)
    topics_sklearn = []
    topics_gensim = []
    topics_tlda = []
    for topic in range(n_tops):
        #top_sklearn = np.argpartition(factors_sklearn[topic],-K)[-K:]
        #top_gensim = np.argpartition(factors_gensim[topic],-K)[-K:]
        #top_tlda = np.argpartition(factors_tlda[topic],-K)[-K:]
        top_sklearn = np.argsort(factors_sklearn[topic])[::-1][:K]
        top_gensim = np.argsort(factors_gensim[topic])[::-1][:K]
        top_tlda = np.argsort(factors_tlda[topic])[::-1][:K]
        #print(top_sklearn)
        
        topics_sklearn.append((vocab)[top_sklearn.astype(int)])
        topics_gensim.append((vocab)[top_gensim.astype(int)])
        topics_tlda.append((vocab)[top_tlda.astype(int)])

        #sklearn_coherence.append(coherence_mean_npmi(top_sklearn, tcm, smooth, n_doc_tcm))
        #sklearn_cosim.append(coherence_mean_npmi_cosim(top_sklearn, tcm, smooth, n_doc_tcm))
        #gensim_coherence.append(coherence_mean_npmi(top_gensim, tcm, smooth, n_doc_tcm))
        #gensim_cosim.append(coherence_mean_npmi_cosim(top_gensim, tcm, smooth, n_doc_tcm))
        #tlda_coherence.append(coherence_mean_npmi(top_tlda, tcm, smooth, n_doc_tcm))
        #tlda_cosim.append(coherence_mean_npmi_cosim(top_tlda, tcm, smooth, n_doc_tcm))
    #print("sklearn")
    #print(sklearn_coherence, sklearn_cosim, np.mean(sklearn_coherence), np.mean(sklearn_cosim))
    #print("gensim")
    #print(gensim_coherence, gensim_cosim, np.mean(gensim_coherence), np.mean(gensim_cosim))
    #print("tlda")
    #print(tlda_coherence, tlda_cosim, np.mean(tlda_coherence), np.mean(tlda_cosim))
    
    pickle.dump(topics_sklearn, open("results/uci_data_sklearn_topics.obj", "wb"))
    pickle.dump(topics_gensim, open("results/uci_data_gensim_topics.obj", "wb"))
    pickle.dump(topics_tlda, open("results/uci_data_tlda_topics.obj", "wb"))
    pickle.dump(texts_lemmatized, open("data/uci_texts_lemmatized.obj", "wb"))
    pickle.dump(id2word, open("data/uci_texts_dictionary.obj", "wb"))

    sklearn_coh = models.coherencemodel.CoherenceModel(topics=topics_sklearn, texts=texts_lemmatized, dictionary = id2word, coherence='u_mass')
    print(sklearn_coh.get_coherence())
    print(topics_gensim)
    gensim_coh = models.coherencemodel.CoherenceModel(topics=topics_gensim, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    print(gensim_coh.get_coherence())
    tlda_coh = models.coherencemodel.CoherenceModel(topics=topics_tlda, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    print(topics_tlda)
    print(tlda_coh.get_coherence())
    return

if __name__ == '__main__':
    main()

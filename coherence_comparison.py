import numpy as np
import cupy as cp
# import scipy
# from scipy.stats import gamma
# from sklearn.decomposition import IncrementalPCA
# from sklearn import preprocessing
# from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation as sklearn_LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Import TensorLy
import tensorly as tl
# from tensorly.tenalg import kronecker
# from tensorly import norm
# from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
# from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
# from tensorly.testing import assert_array_equal, assert_array_almost_equal
# from cumulant_gradient import cumulant_gradient
# import pandas as pd 

import time
# import csv

#Insert Plotly
# import matplotlib.pyplot as plt
import pickle
# Import utility functions from other files
from version0_99.tlda import TLDA
from version0_99.pca import PCA
# import version0_20.test_util as test_util
# import version0_20.tensor_lda_util as tl_util
from version0_99.preprocess_efficient import cleanLine, regexchars, tokenize, removeStopwords, stem

# import version0_15.tensor_lda_clean as tlda_mid
#import version0_20.tlda_final_validation as tlda_final_validation

import gensim.corpora as corpora
import gensim.models as models
# from gensim.test.utils import datapath

#backend="cupy"
#backend = "numpy"
#tl.set_backend(backend)

VOCAB = 1000

# def coherence_mean_npmi (term_indices, tcm, smooth, n_doc_tcm):
#     #given suitably ordered pairs of indices stored in two column matrix "indices" a non-vectorized calculation would be something like
#     #mapply(function(x, y)  {(log2((tcm[x,y]/n_doc_tcm) + smooth) - log2(tcm[x,x]/n_doc_tcm) - log2(tcm[y,y]/n_doc_tcm)) / -log2((tcm[x,y]/n_doc_tcm) + smooth)}}    #                        , indices[,1], indices[,2])
#     tl.set_backend('numpy')
#     if n_doc_tcm <= 0:
#         return 0
#     res = None
#     n = len(term_indices)
#     if(n >= 2):
#         res = tcm[np.ix_(term_indices, term_indices)] / n_doc_tcm
#         res[np.triu_indices(n, k=1)] = res[np.triu_indices(n, k=1)] + smooth
#         # interim storage of a denominator
#         denominator =  res[np.triu_indices(n, k=1)]
#         d = np.diag(res)
#         res = (res.T/d).T
#         res = np.dot(res, np.diag(1 / d))
#         res = res[np.triu_indices(n, 1)]
#         res = np.log2(res) / -np.log2(denominator)
#         res = np.mean(res, where = ~np.isnan(res))
#     return res

# def coherence_mean_npmi_cosim(term_indices, tcm, smooth, n_doc_tcm):
#     #TODO
#     #example of nonvectorized calculation
#     tl.set_backend('numpy')
#     if n_doc_tcm <= 0:
#         return 0
#     res = None
#     n = len(term_indices)
#     if n >= 2:
#         res = tcm[np.ix_(term_indices, term_indices)] / n_doc_tcm
#         res[np.tril_indices(n, k = -1)] = (res.T)[np.tril_indices(n, k = -1)]
#         res = res + smooth
#         res[np.diag_indices(n)] = np.diag(res) - smooth
#         #interim storage of denominator
#         denominator =  res
#         d = np.diag(res)
#         res = (res.T/d).T
#         res = np.dot(res, np.diag(1 / d))
#         res = np.log2(res) / -np.log2(denominator)
#         #create values for cosine similarity check, for this metric: the sum of all npmi values
#         res_compare = (np.reshape(np.tile(np.ndarray.sum(res, axis=0), n), (n, n))).T
#         res_norm = preprocessing.normalize(res, norm='l2')
#         res_compare_norm = preprocessing.normalize(res_compare, norm='l2')
#         res = metrics.pairwise.cosine_similarity(res_norm, res_compare_norm)
#         res = np.mean(res, where = ~np.isnan(res))
#     return(res)

def get_uci_data():
    #tl.set_backend('numpy')
    with tl.backend_context('numpy'):
        test_len = None
        #test_len = 20
    
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
        vectorizer = CountVectorizer(max_df = 0.1, min_df = 0.02)
        vectors = vectorizer.fit_transform(texts).toarray()
        vocab = vectorizer.get_feature_names()
        print(len(vocab))
        print(len(vectorizer.stop_words_))
    
        texts2 = []
        for text in texts:
            texts2.append([w for w in text.split(' ') if w.lower() in vectorizer.vocabulary_])

        return vectors, vocab, texts2

# def get_congress_data():
#     with tl.backend_context('numpy'):
#         df = pd.read_csv("data/TwitterCong_Jul2021.csv")
#         tweets = df['tweet']
#         tweets = [stem(removeStopwords(tokenize(regexchars(cleanLine(line))))) for line in tweets]
#         pickle.dump(tweets, open("data/preprocessed_congress_tweets.obj", 'wb'))
#         vectorizer = CountVectorizer(max_df = 0.1, min_df = 0.0001)
#         vectors = vectorizer.fit_transform(tweets).toarray()
#         pickle.dump(vectors, open("data/countvecs_congress_tweets.obj", 'wb'))
#         vocab = vectorizer.get_feature_names()
#         pickle.dump(vocab, open("data/congress_vocab.obj", 'wb'))
#     return vectors, vocab

# def get_metoo_data():
#     with tl.backend_context('numpy'):
#         #df = pd.read_csv("data/MeTooMonthCleaned/twitter_per_month_201701.csv", header=0, names=["tweets"], dtype = str)
#         #print(df.head())
#         #tweets = df['tweets'].tolist()
#         #print(tweets[0:10])
#         #tweets = [stem(removeStopwords(tokenize(regexchars(cleanLine(str(line)))))) for line in tweets]
#         #tweets = pickle.load(open("data/preprocesses.obj", 'rb'))
#         #pickle.dump(tweets, open("data/preprocessed_metoo_tweets.obj", 'wb'))
#         #vectorizer = CountVectorizer(max_df = 0.01, min_df = 0.0001)
#         #vectors = vectorizer.fit_transform(tweets).toarray()
#         #pickle.dump(vectors, open("data/countvecs_metoo_tweets.obj", 'wb'))
#         #vocab = vectorizer.get_feature_names()
#         #print(len(vocab))
#         #pickle.dump(vocab, open("data/metoo_vocab.obj", 'wb'))
#         vectors = pickle.load(open("data/Meena_testing/x_mat/twitter_per_month_201701.obj", 'rb'))
#         vectorizer = pickle.load(open("data/Meena_testing/countvec.obj", "rb"))
#         vocab = vectorizer.get_feature_names()
#         print(vocab)
#     return vectors, vocab

# def postprocess(factors_unwhitened, x, vocab, num_tops, smoothing, decenter=False, name="", alpha_0 = 1):
#     '''Post-Processing '''
#     res = []
#     # Postprocessing
#     tl.set_backend("numpy")
#     #This is hard-coded. We should calculate the alphas by hand. 
#     if decenter == True:
#         #eig_vals = cp.array([np.linalg.norm(k)**3 for k in factors_unwhitened.T ])
#         # normalize beta
#         #alpha           = cp.power(eig_vals, -2)
#         #alpha_norm      = (alpha / alpha.sum()) * alpha_0
#         #weights   = tl.tensor(alpha_norm)
#         #print("weights shape:")
#         #print(weights.shape)

#         #fac2 = factors_unwhitened/weights
#         #print("fac2 shape: ")
#         #print(fac2.shape)
#         #fac2 = (fac2.T + tl.mean(x, axis=0)).T
#         #fac2 *= weights
#         #fac2 = cp.asarray(fac2)
#         #print("final fac2: ")
#         #print(fac2)

#         #print("decenter with new strategy:")
#         #print(fac2[0])
#         t1 = time.time()
#         #wc   =  cp.asarray(tl.mean(x, axis=0))#/vocab*(1/num_tops)
#         wc = np.asarray(tl.mean(x, axis=0))
#         wc   =  tl.reshape(wc,(vocab,1))
        
#         factors_unwhitened = cp.asnumpy(factors_unwhitened)
#         #factors_unwhitened   =  cp.asarray(factors_unwhitened)
#         factors_unwhitened += wc
#         t2 = time.time()
#         print("Decentering: " + str(t2-t1))
#         res.append((name + ' decentering', t2-t1))
#         print("decenter with old strategy:")
#         print(factors_unwhitened[0])

#     #print(factors_unwhitened.dtype)
#     #print(wc.dtype)
#     #print(factors_unwhitened.shape)
#     #print(wc.shape)

#     #factors_unwhitened   =  cp.asarray(factors_unwhitened)
#     factors_unwhitened = np.asarray(factors_unwhitened)
#     # print(factors_unwhitened)
#     t1 = time.time()
#     factors_unwhitened [factors_unwhitened  < 0.] = 0.
#     # smooth beta
#     factors_unwhitened  *= (1. - smoothing)
#     #print(factors_unwhitened)

#     factors_unwhitened += (smoothing / factors_unwhitened.shape[1])
#     #print(factors_unwhitened)
#     #print("begin print estimated mu")
#     factors_unwhitened /= factors_unwhitened.sum(axis=0)
#     t2 = time.time()
#     print("Smoothing and Normalization: " + str(t2-t1))
#     res.append((name + ' smoothing and normalization', t2-t1))
#     #print(factors_unwhitened)
#     # remean the data
#     #print("begin mean")
#     #if decenter == True:
#     #    fac2[fac2 < 0.] = 0.
#     #    fac2 *= (1. - smoothing)
#     #    fac2 += (smoothing/fac2.shape[1])
#     #    fac2 /= fac2.sum(axis=0)
#     # print(wc)
#     # print("begin ground truth")
#     # print(mu)


#     """ INSERT CODE FOR COHERENCE HERE """
#     # if decenter == True:
#     #    return res, fac2
#     return res, factors_unwhitened


# def gen_fit_0_15(x, num_tops = 2, alpha_0 = 0.01, n_iter_max=200, theta=1, learning_rate = 0.01, seed=None):
#     num_tops = num_tops
#     vocab   = x.shape[1]
#     theta = theta
#     learning_rate = learning_rate
#     #seed = seed
#     smoothing  =  1e-5#0.001 #1e-5 #1e-5

#     res = []
    
#     #backend="cupy"
#     backend="numpy"
#     tl.set_backend(backend)
#     #mempool = cp.get_default_memory_pool()
#     #mempool.free_all_blocks()
#     #pinned_mempool = cp.get_default_pinned_memory_pool()
#     #pinned_mempool.free_all_blocks()
    
#     x = tl.tensor(x)

#     t1 = time.time()
#     M1 = tlda_mid.get_M1(x)
#     t2 = time.time()
#     print("M1: " + str(t2-t1))
#     res.append(('M1 calc', t2-t1))

#     t1 = time.time()
#     M2_img = tlda_mid.get_M2(x, M1, alpha_0)
#     t2 = time.time()
#     print("M2: " + str(t2-t1))
#     res.append(('M2 calc', t2-t1))

#     t1 = time.time()
#     W, W_inv = tlda_mid.whiten(M2_img, num_tops) # W (n_words x n_topics)
#     t2 = time.time()
#     print(tl.dot(tl.dot(W.T, M2_img), W))
#     print("W: " + str(t2-t1))
#     res.append(('W calc', t2-t1))

#     W = tl.tensor(W)
#     W_inv = tl.tensor(W_inv)
#     print(tl.context(W))
#     print(tl.context(W_inv))

#     t1 = time.time()
#     X_whitened = tl.dot(x, W)
#     t2 = time.time()
#     print("Whiten X: " + str(t2-t1))
#     res.append(('whiten X', t2-t1))
#     print(tl.get_backend())
#     res_copy = res.copy()
#     print(tl.context(X_whitened))

#     # This is where the two versions branch off -- begin with version 0.10
#     t1 = time.time()
#     M1_whitened = tl.dot(M1, W)
#     t2 = time.time()
#     print("Whiten M1: " + str(t2-t1))
#     res.append(('whiten M1', t2-t1))

#     t1 = time.time()
#     M3_final = tlda_mid.get_M3(X_whitened, M1_whitened, alpha_0)
#     t2 = time.time()
#     print("Parafac M3: " + str(t2-t1))
#     res.append(('construct M3', t2-t1))
#     #print((M3_final).device)

#     print(tl.get_backend())
#     t1 = time.time()
#     lambdas_learned_parafac, phis_learned_parafac = sym_parafac(M3_final, rank=num_tops, n_repeat=10, n_iteration=100, verbose=False)
#     t2 = time.time()
#     print("Parafac Decomposition: " + str(t2-t1))
#     res.append(('decompose parafac', t2-t1))
#     print(tl.get_backend())
#     t1 = time.time()
#     factors_unwhitened_parafac     = (tl.dot(W_inv,phis_learned_parafac )) 
#     t2 = time.time()
#     print("Unwhitening parafac factors: " + str(t2-t1))
#     res.append(('unwhiten factors parafac', t2-t1))

#     t1 = time.time()
#     #weights, phis_learned  = tlda_mid.simulate_all(X_whitened, alpha_0, num_tops, lr1 = learning_rate, theta=theta, seed=seed, verbose = False,min_iter = 10,max_iter=100)#n_iter_max)
#     t2 = time.time()
#     print("SGD Calc: " + str(t2-t1))
#     res_copy.append(('SGD calc', t2-t1))

#     t1 = time.time()
#     #factors_unwhitened     = (tl.dot(W_inv,phis_learned )) 
#     t2 = time.time()
#     print("Unwhitening factors: " + str(t2-t1))
#     res_copy.append(('unwhiten factors SGD', t2-t1))

#     res3, factors_parafac = postprocess(factors_unwhitened_parafac, x, vocab, num_tops, smoothing, decenter=False, name="parafac")
#     # res3 = {}
#     # res2 = []
#     # accuracy_uncentered = None
#     #res2, factors_uncentered = postprocess(factors_unwhitened, x, vocab, num_tops, smoothing, decenter=False)
#     # {**{**res, **res2}, **res3}
#     res.extend(res3)
#     #res_copy.extend(res2)
#     return res, res_copy, factors_parafac, []#factors_uncentered 

def gen_fit_0_20(x, num_tops = 2, alpha_0 = 0.01, n_iter_train = 2001):
    vocab   = x.shape[1]
    n_iter_train     = n_iter_train
    batch_size_pca = x.shape[0] # 100000 # 2000
    batch_size_grad  = 100
    n_iter_test = 10 
    theta_param =  10
    learning_rate = 0.0001
    smoothing  = 1e-5 #1e-5

    # res = {}
    res = []

    # backend="cupy"
    backend = "numpy"
    tl.set_backend(backend)
    
    x = tl.tensor(x)


    t1 = time.time()
    M1 = tl.mean(x, axis=0)
    x_cent = x - M1
    t2 = time.time()
    print("Centering time: " + str(t2-t1))
    res.append(('centering', t2-t1))

    t1 = time.time()
    pca = PCA(num_tops, alpha_0, batch_size_pca,backend)
    pca.fit(x_cent)
    t2 = time.time()
    print("PCA fit: " + str(t2-t1))
    res.append(('PCA fit', t2-t1))
    #print(tl.get_backend())
    # pickle.dump(pca, open("data/pca_metoo_cpu.obj", "wb"))
    #pca = pickle.load(open("data/pca_metoo_cpu.obj", "rb"))
    # M2_img = tlda_mid.get_M2(x_cent, tl.mean(x_cent, axis=0), alpha_0)
    # W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    # print(tl.dot(tl.dot(W.T, M2_img), W))
    #print(tl.context(x_cent))

    t1 = time.time()
    # W = pca.projection_weights_ / tl.sqrt(pca.whitening_weights_)[None, :]
    x_whit = pca.transform(x_cent)
    t2 = time.time()
    print("PCA Transform: " + str(t2-t1))
    res.append(('PCA transform', t2-t1))
    print(tl.context(x_cent))

    '''fit the tensor lda model'''
    t1 = time.time()
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,gamma_shape = 1.0, smoothing = smoothing,theta=theta_param)
    tlda.fit(x_whit,verbose=False)
    t2 = time.time()
    print(tl.get_backend())
    print("TLDA fit: " + str(t2-t1))
    res.append(('TLDA fit', t2-t1))
    # pickle.dump(tlda, open("data/tlda_metoo_cpu.obj", "wb"))
    #print(tlda.factors_.device)

    t1 = time.time()
    factors_unwhitened = tlda.postprocess(pca, M1, vocab)
    t2 = time.time()
    #print(factors_unwhitened.device)
    print("postprocess : " + str(t2-t1))
    res.append(('postprocess', t2-t1))
    
    # res2, factors_unwhitened = postprocess(factors_unwhitened, x, vocab, num_tops, smoothing, True, alpha_0 = alpha_0)
    #tl.mean(cp.asnumpy(accuracy))
    # {**res, **res2}
    # res.extend(res2)
    # print(tl.get_backend())
    return res, factors_unwhitened

def output_all_coherences(topics, texts, dict):
    coh_dict = {}
    for name in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
        coh = models.coherencemodel.CoherenceModel(topics=topics, texts=texts, dictionary = dict, coherence=name)
        coh_dict[name] = coh.get_coherence()
    return coh_dict

def main():
    n_tops = 20
    alpha_0 = 0.01
    tl.set_backend("numpy") 
    print("starting")
    texts, vocab, texts_lemmatized = get_uci_data()
    #texts, vocab = get_congress_data()
    #texts, _ = get_metoo_data()
    
    #df = pd.read_csv("data/MeTooMonthCleaned/twitter_per_month_201701.csv", header=0, names=["tweets"], dtype = str)
    #print(df.head())
    #texts_raw = df['tweets'].tolist()
    #print(tweets[0:10])
    #texts_raw = [stem(removeStopwords(tokenize(regexchars(cleanLine(str(line)))))) for line in texts_raw]
    
    #return
    #texts = pickle.load( open('data/countvecs_congress_tweets.obj', 'rb'))
    #vectorizer = pickle.load(open("data/Meena_testing/countvec.obj", "rb"))
    #vocab = cp.asnumpy(vectorizer.get_feature_names())
    #df = pd.read_csv("data/vocab.csv")
    #vocab = df['words'].tolist()
    #texts_raw = pickle.load(open("data/preprocessed_metoo_tweets.obj", "rb"))
    #texts_lemmatized = []
    ##texts_raw = texts
    #for text in texts_raw:
    #    texts_lemmatized.append([w for w in text.split(' ') if w.lower() in vocab])
    #pickle.dump(texts_lemmatized, open("data/countvecs_metoo_lemmatized.obj", 'wb'))
    #return
    #texts_lemmatized = pickle.load(open("data/countvecs_metoo_lemmatized.obj", 'rb'))
    #print("got data")
    #print(texts.shape)
    

    #tcm = tl.dot(texts.T, texts)

    t1 = time.time()
    lda = sklearn_LDA(n_components = n_tops)
    lda.fit(texts)
    factors_sklearn = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    t2 = time.time()
    res2 = ('sklearn LDA', t2 - t1)
    print(res2)
    
    id2word = corpora.Dictionary(texts_lemmatized)
    corpus = [id2word.doc2bow(text) for text in texts_lemmatized]
    t1 = time.time()
    lda_model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=n_tops,
                                       #chunksize=100,
                                    #    passes=10,
                                       #alpha_0,
                                       per_word_topics=False)
    factors_gensim = lda_model.get_topics()
    t2 = time.time()
    res3 = ('gensim LDA', t2 - t1)
    print(res3)
    #print(res3)
    #pickle.dump(factors_gensim, open("data/gensim_metoo_factors.obj", "wb"))
    #factors_gensim = pickle.load(open("data/gensim_metoo_factors.obj", "wb"))
    #temp_file = datapath("data/gensim_metoo_model")
    #lda_model.save(temp_file)
    
    res, factors_tlda = gen_fit_0_20(texts, num_tops = n_tops, alpha_0 = alpha_0, n_iter_train = 10001)
    outFile = open("results/res_tlda.txt", "w")
    print(res, file=outFile)
    print(res2, file=outFile)
    print(res3, file=outFile)
    outFile.close()
    #pickle.dump(factors_tlda, open("data/tlda_factors_cpu_metoo.obj", "wb"))
    #print(factors_tlda)
    #factors_tlda = factors_tlda.T
    #print(factors_tlda.shape)
    #print(res)
    #factors_tlda = pickle.load(open("data/Meena_testing/tlda_factors_metoo.obj", "rb"))
    #factors_tlda = factors_tlda.T
   # factors_tlda = cp.asnumpy(factors_tlda).T
    #return
    #print(factors_tlda.device)
    #print(tl.get_backend())
    #tl.set_backend('numpy')
    #tl.set_backend('cupy')
    #tlda_coh = models.coherencemodel.CoherenceModel(topics=topics_tlda, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    #print(topics_tlda)
    #print(tlda_coh.get_coherence())

    #res_parafac, res_uncentered, factors_parafac, factors_uncentered = gen_fit_0_15(texts, num_tops=n_tops, alpha_0 = alpha_0, theta=1, learning_rate = 0.001)
    #factors_parafac = factors_parafac.T
    #factors_uncentered = factors_uncentered.T
    
    #tl.set_backend('numpy')
    K = 20
    vocab = np.asarray(vocab)
    #texts = cp.asnumpy(texts)
    #texts_lemmatized = cp.asnumpy(texts_lemmatized)
    # tcm = tl.mean(batched_tensor_dot(texts, texts), axis=0)
    #print(tcm.shape)
    #n_doc_tcm = len(texts)
    #smooth = 0.0001

    sklearn_coherence  = []
    gensim_coherence = []
    tlda_coherence = []
    sklearn_cosim = []
    gensim_cosim = []
    tlda_cosim = []
    
    factors_tlda = cp.asnumpy(factors_tlda)
    #factors_parafac = cp.asnumpy(factors_parafac)
    #factors_uncentered = cp.asnumpy(factors_uncentered)

    #print(factors_sklearn.shape)
    #print(factors_gensim.shape)
    #print(factors_tlda.shape)

    #factors_tlda = cp.asnumpy(factors_tlda)
    topics_sklearn = []
    topics_gensim = []
    topics_tlda = []
    # topics_parafac = []
    # topics_uncentered = []
    for topic in range(n_tops):
        #top_sklearn = np.argpartition(factors_sklearn[topic],-K)[-K:]
        #top_gensim = np.argpartition(factors_gensim[topic],-K)[-K:]
        top_tlda = np.argpartition(factors_tlda[topic],-K)[-K:]
        top_sklearn = np.argsort(factors_sklearn[topic])[::-1][:K]
        top_gensim = np.argsort(factors_gensim[topic])[::-1][:K]
        #top_tlda = np.argsort(factors_tlda[topic])[::-1][:K]
        #print(top_tlda)
        
        #top_parafac = np.argsort(factors_parafac[topic])[::-1][:K]
        #top_uncentered = np.argsort(factors_uncentered[topic])[::-1][:K]
        #print(top_sklearn)
        
        topics_sklearn.append((vocab)[top_sklearn.astype(int)])
        topics_gensim.append((vocab)[top_gensim.astype(int)])
        topics_tlda.append((vocab)[top_tlda.astype(int)])
        #topics_parafac.append((vocab)[top_parafac.astype(int)])
        #topics_uncentered.append((vocab)[top_uncentered.astype(int)])

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
    
    #pickle.dump(topics_sklearn, open("results/uci_data_sklearn_topics.obj", "wb"))
    #pickle.dump(topics_gensim, open("results/uci_data_gensim_topics.obj", "wb"))
    #pickle.dump(factors_tlda, open("results/metoo_data_tlda_topics.obj", "wb"))
    #pickle.dump(topics_parafac, open("results/uci_data_parafac_topics2.obj", "wb"))
    #pickle.dump(topics_uncentered, open("results/uci_data_uncentered_topics.obj", "wb"))
    #pickle.dump(texts_lemmatized, open("data/uci_texts_lemmatized.obj", "wb"))
    #pickle.dump(id2word, open("data/uci_texts_dictionary.obj", "wb"))
    #outFile=open("results/tlda_topics_metoo.txt", 'w')
    #print(topics_tlda, file=outFile)
    #outFile.close()
    sklearn_coh = output_all_coherences(topics_sklearn, texts_lemmatized, id2word)
    outFile = open("results/sklearn_coherence.txt", "w")
    print(sklearn_coh, file=outFile)
    outFile.close()

    gensim_coh = output_all_coherences(topics_gensim, texts_lemmatized, id2word)
    outFile = open("results/gensim_coherence.txt", "w")
    print(gensim_coh, file=outFile)
    outFile.close()

    tlda_coh = output_all_coherences(topics_tlda, texts_lemmatized, id2word)
    outFile = open("results/tlda_coherence.txt", "w")
    print(tlda_coh, file=outFile)
    outFile.close()

    #print(topics_gensim)
    # gensim_coh = models.coherencemodel.CoherenceModel(topics=topics_gensim, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    # print(gensim_coh.get_coherence())
    #tlda_coh = models.coherencemodel.CoherenceModel(topics=topics_tlda, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    #print(topics_tlda)
    #print(tlda_coh.get_coherence())
    #parafac_coh = models.coherencemodel.CoherenceModel(topics=topics_parafac, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    #print(parafac_coh.get_coherence())
    #uncentered_coh = models.coherencemodel.CoherenceModel(topics=topics_uncentered, texts=(texts_lemmatized), dictionary = id2word, coherence='u_mass')
    #print(uncentered_coh.get_coherence())
    return

if __name__ == '__main__':
    main()

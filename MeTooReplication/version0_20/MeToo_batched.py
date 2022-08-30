import numpy as np
import cupy as cp
import scipy
import os
from os.path import isfile, join

# Import stopwords
import nltk
#from nltk import word_tokenize
from nltk.corpus import stopwords
# Import TensorLy
import tensorly as tl
import cudf
from   cudf import Series
from cuml.feature_extraction.text import CountVectorizer
from cuml.preprocessing.text.stem import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem import PorterStemmer
import gc

#Insert Plotly
#import matplotlib.pyplot as plt
import pandas as pd
# Import utility functions from other files
from tlda_final import TLDA
from pca import PCA
import test_util
import tensor_lda_util as tl_util
import time
import pickle

# class StemTokenizer(object):
#     def __init__(self):
#         self.porter = PorterStemmer()
#     def __call__(self, articles):
#         return [self.porter.stem(t) for t in word_tokenize(articles)]



backend="cupy"
tl.set_backend(backend)
device = 'cuda'#cuda

porter = PorterStemmer()


    


def basic_clean(df):
    df['tweets'] = df['tweets'].astype('str')
    df = df.drop_duplicates(keep="first")
    return df

def partial_fit(self , data):
    if(hasattr(self , 'vocabulary_')):
        vocab = self.vocabulary_ # series
    else:
        vocab = Series()
    self.fit(data)
    vocab = vocab.append(self.vocabulary_)
    self.vocabulary_ = vocab.unique()



stop_words = (stopwords.words('english'))
added_words = ["thread","say","will","has","by","for","hi","hey","hah"
               "said","talk","congrats","congratulations","are","as","i", 
               "me", "my", "myself", "we", "our", "ours", "ourselves", 
               "you", "your", "yours","he","her","him","she","hers","that",
               "be","with","their","they're","is","was","been","not","they",
               "it","have",  "one","think",   "thing"    ,"bring","put","well",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"things","can't","can",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha",
               "lets","let's","say","says","know","talk","talked","talks","dont","think",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat",
               "something","wont","people","make","want","went","goes","people","had",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah",
               "need","us", "men", "women", "get", "woman", "man", "amp","amp&","yr","yrs"]



stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit

countvec = CountVectorizer( stop_words = stop_words, # works
                            lowercase = True, # works
                            ngram_range = (1,2),
                            max_df = 800000, # works
                            min_df = 2000)


inDir  = "../data/MeTooMonthCleaned/" # input('Name of input directory? : ')

num_tops = 15
alpha_0 = 0.01
batch_size_pca = 20000
batch_size_grad = 8000
n_iter_train    = 2000
n_iter_test     = 1
learning_rate = 0.0001
theta_param   = 2
smoothing     = 0.000000001

# Program controls
vocab_build = 1
save_files  = 1
pca_run     = 1
whiten      = 1
stgd        = 1

#Start

print("\n\nSTART...")
dir_contents = os.listdir(inDir)
dl = [x for x in dir_contents if isfile(join(inDir, x))]

if vocab_build == 1:
    for f in dl:
        print("Beginning vocabulary build: " + f)
        path_in  = os.path.join(inDir,f)
        # read in dataframe 
        df = cudf.read_csv(path_in, names = ['tweets'])

        # basic preprocessing
        df = basic_clean(df)
        countvec.partial_fit(df['tweets'])
        print("End " + f)


    vocab = len(countvec.vocabulary_)
    M1_sum = tl.zeros(vocab)
    tot_len = 0
    for f in dl:
        print("Beginning transform/mean: " + f)
        path_in  = os.path.join(inDir,f)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        # read in dataframe 
        df = pd.read_csv(path_in, names = ['tweets'])
        mask = df['tweets'].str.len() > 2 
        df   = df.loc[mask]
        df   = cudf.from_pandas(df)
        # basic preprocessing
        df   = basic_clean(df)

        X_batch = tl.tensor(countvec.transform(df['tweets']).toarray())
        M1_sum += tl.sum(X_batch, axis=0)
        print(X_batch.shape[1])
        tot_len += X_batch.shape[0]
        print(str(tot_len))
        if save_files == 1:
            pickle.dump(X_batch, open('../data/x_mat/' + f[:-4] + '.obj','wb'))
        print("End " + f)

    M1 = M1_sum/tot_len
    print("Total length of dataset: " + str(tot_len))

    pickle.dump(countvec, open('../data/countvec.obj','wb'))
    pickle.dump(M1, open('../data/M1.obj','wb'))
    del M1_sum
    del X_batch 
    del df
    del mask
    gc.collect()

if vocab_build ==0:
    countvec = pickle.load(open('../data/countvec.obj','rb'))
    M1       = pickle.load(open('../data/M1.obj','rb'))
    vocab = len(countvec.vocabulary_)



gc.collect()

pca = PCA(num_tops, alpha_0, batch_size_pca,backend)

if pca_run == 1:
    t1 = time.time()
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print("Beginning PCA: " + f)
        X_batch = pickle.load( open('../data/x_mat/' + f[:-4] + '.obj','rb'))
        X_batch -= M1 # center the data

        for j in range(0, len(X_batch)-(batch_size_pca-1), batch_size_pca):
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            y = X_batch[j:j+batch_size_pca]
            pca.partial_fit(y)

    t2 = time.time()
    print("PCA and Centering Time: " + str(t2-t1))
    pickle.dump(pca, open('../data/pca.obj','wb'))
    pickle.dump(pca.projection_weights_, open('../data/pca_proj_weights.obj','wb'))
    pickle.dump(pca.whitening_weights_, open('../data/pca_whitening_weights.obj','wb'))
    del X_batch 
    del y

gc.collect()
if pca_run ==0:
    pca = pickle.load(open('../data/pca.obj','rb'))

gc.collect()
tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,gamma_shape = 1.0, smoothing = 1e-6,theta=theta_param)
if whiten == 1:
    t1 = time.time()
    x_whits = []
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print("Beginning TLDA: " + f)
        X_batch = pickle.load( open('../data/x_mat/' + f[:-4] + '.obj','rb'))
        X_batch -= M1
        x_whits.append(pca.transform(X_batch))
    
    x_whit = tl.concatenate(x_whits, axis=0)
    print(x_whit.shape)
    pickle.dump(x_whit, open('../data/x_whit.obj','wb'))
    t2 = time.time()
    print("Whiten time: " + str(t2-t1))

if whiten == 0:
    x_whit=pickle.load(open('../data/x_whit.obj','rb'))
gc.collect()
if stgd == 1:
    t1 = time.time()
    tlda.fit(x_whit,verbose=True)
    t2 = time.time()

    print("TLDA Time: " + str(t2-t1))

    t1 = time.time()
    factors_unwhitened = pca.reverse_transform(tlda.factors_)
    factors_unwhitened = factors_unwhitened.T
    t2 = time.time()
    print("Unwhitening Time: " + str(t2-t1))
    pickle.dump(factors_unwhitened, open('../data/preprocess_factors_MeToo.obj', 'wb'))
    pickle.dump(tlda, open('../data/tlda.obj', 'wb'))

if stgd == 0:
        factors_unwhitened = pickle.load(open('../data/preprocess_factors_MeToo.obj', 'rb'))
        tlda               = pickle.load(open('../data/tlda.obj', 'rb'))


'''Post-Processing '''

eig_vals = cp.array([np.linalg.norm(k)**3 for k in tlda.factors_ ])
# normalize beta
alpha           = cp.power(eig_vals, -2)
alpha_norm      = (alpha / alpha.sum()) * alpha_0
tlda.weights_   = tl.tensor(alpha_norm)
print(tlda.weights_)


# We need to fix this bit...
wc   =  cp.asarray(M1)/vocab*(1/num_tops) #tlda.weights_
wc   =  tl.reshape(wc,(vocab,1))


t1 = time.time()
factors_unwhitened   =  cp.asarray(factors_unwhitened)
factors_unwhitened += wc


factors_unwhitened[factors_unwhitened  < 0.] = 0.
factors_unwhitened  *= (1. - smoothing)

factors_unwhitened += (smoothing / factors_unwhitened.shape[1])
factors_unwhitened /= factors_unwhitened.sum(axis=0)


t2 = time.time()

pickle.dump(factors_unwhitened, open('../data/learned_factors_MeToo.obj', 'wb'))

n_top_words = 15
df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
df_voc.to_csv('../data/vocab.csv')

for k in range(0,num_tops):
    if k ==0:
        t_n_indices   = factors_unwhitened[:,k].argsort()[:-n_top_words - 1:-1]
        top_words_LDA = countvec.vocabulary_[t_n_indices]
        top_words_df  = cudf.DataFrame({'words_'+str(k):top_words_LDA}).reset_index(drop=True)
        
    if k >=1:
        t_n_indices   = factors_unwhitened[:,k].argsort()[:-n_top_words - 1:-1]
        top_words_LDA = countvec.vocabulary_[t_n_indices]
        top_words_df['words_'+str(k)] = top_words_LDA.reset_index(drop=True)
        print(top_words_df.head())

       
top_words_df.to_csv('../data/top_words.csv')
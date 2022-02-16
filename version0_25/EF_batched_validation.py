import numpy as np
import cupy as cp
import scipy
import os
from os.path import exists, isfile, join
from pathlib import Path
import shutil
import gc
import math


# Import stopwords
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

# Import TensorLy
import tensorly as tl
import cudf
from cudf import Series
from cuml.feature_extraction.text import CountVectorizer
from cuml.preprocessing.text.stem import PorterStemmer
import cupyx 
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem import PorterStemmer

#Insert Plotly
#import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

# Import utility functions from other files
from tlda_final_validation import TLDA
from pca   import PCA
import tensor_lda_mid as tlda_mid
import test_util_validation
import tensor_lda_util as tl_util
import file_operations as fop

# class StemTokenizer(object):
#     def __init__(self):
#         self.porter = PorterStemmer()
#     def __call__(self, articles):
#         return [self.porter.stem(t) for t in word_tokenize(articles)]


# Constants
X_MAT_FILEPATH_PREFIX = '../data/x_mat/'
COUNTVECTOR_FILEPATH = '../data/countvec.obj'
M1_FILEPATH = '../data/M1.obj'
PCA_FILEPATH = '../data/pca.obj'
PCA_PROJ_WEIGHTS_FILEPATH = '../data/pca_proj_weights.obj'
PCA_WHITENING_WEIGHTS_FILEPATH = '../data/pca_whitening_weights.obj'
X_WHITENED_FILEPATH = '../data/x_whit.obj'
TLDA_FILEPATH = '../data/tlda.obj'
PREPROCESS_FACTORS_METOO_FILEPATH = '../data/preprocess_factors_election.obj'
POST_FACTORS_METOO_FILEPATH       = '../data/postprocess_factors_MeToo.obj' 
TOP_WORDS_FILEPATH = '../data/top_words.csv'
VOCAB_FILEPATH = '../data/vocab.csv'
TOTAL_DATA_ROWS_FILEPATH = '../data/total_data_rows.obj'
PCA_WHITENED_XMAT_FILEPATH = "../data/xwhiten/"
X_PATH = "../data/x_path/"
WEIGHTS_FILEPATH = "../data/alpha_weights.txt"

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
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
        vocab = vocab.append(Series("stop steal"))
        vocab = vocab.append(Series("mail ballots"))
        vocab = vocab.append(Series("voter suppression"))
        vocab = vocab.append(Series("illegal votes"))
        vocab = vocab.append(Series("trump won"))
        vocab = vocab.append(Series("biden won"))
        vocab = vocab.append(Series("fake news"))
        vocab = vocab.append(Series("election fraud"))
        vocab = vocab.append(Series("united states"))
    self.fit(data)
    vocab = vocab.append(self.vocabulary_)
    
    self.vocabulary_ = vocab.unique()

def tune_filesplit_size_on_IPCA_batch_size(IPCA_batchsize):
    return None


stop_words = (stopwords.words('english'))
added_words = ["thread","say","will","has","by","for","hi","hey","hah","thank","metoo","watch","sexual","doe","biden",
               "said","talk","congrats","congratulations","are","as","i", "time","abus","year","mani","trump","0 ","000",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement","assault","100","united","states",
               "you", "your", "yours","he","her","him","she","hers","that","harass","whi","feel","say","gt","ballots","mail",
               "be","with","their","they're","is","was","been","not","they","womensmarch","way","thi","rigged","evidence","supression",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell","suprresion",
               "good","day","work", "latest","today","becaus","peopl","via","see","timesup","old","ani","realdonaldtrump","ballot",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"thing","can't","can","right","got","show","happened",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back","president",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right"," 0",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen"," 0 ",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need","us", "men", "women", "get", "woman", "man", "amp","amp&","yr","yrs",'voter', 'fraud','election',"states",
               "https","co","http","votes","voters", "vote", "2020", "voters",'0',"00","80","000 000","suppression","state"
         ]



stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit

countvec = CountVectorizer( stop_words = stop_words, # works
                            lowercase = True, # works
                            ngram_range = (1,2),
                            max_df = 15000, # works
                            min_df = 280)


inDir  = "../data/data_split" # input('Name of input directory? : ')

# Learning parameters
num_tops = 50 # 50 topics :(931, 93, 1258) coherence: 2277 (lr=0.00003 )
alpha_0 = 0.1
batch_size_pca = 75000                                                                                                                                                           
batch_size_grad = 6000
n_iter_train    = 1000
n_iter_test     = 1
learning_rate   = 0.00005 
theta_param = 5000.005
ortho_loss_param =50000
smoothing   = 1e-15

# Program controls   
split_files = 0
vocab_build = 1
transform_mean = 1
save_files  = 1
pca_run     = 1
whiten      = 1
stgd        = 1
coherence   = 0

# Other globals
num_data_rows = 0

#Start

print("\n\nSTART...")

# Get a list of files in the directory
#dl = os.listdir(inDir)
dl = fop.get_files_in_dir(inDir)

print(dl)
# Split datafiles into smaller files
print("Splitting files")

if split_files == 1:
    inDir = fop.split_files(
        inDir, 
        os.path.join(
            "EFBatched_clean", 
            "split_files")
    )
    dl = fop.get_files_in_dir(inDir)
    print("Done. Split files located at: {}.\n".format(inDir))
    print("Split files and their filesizes: ")
    fop.print_filesizes(inDir)
    #fop.print_num_rows_in_csvs(inDir)


# Build the vocabulary
if vocab_build == 1:
    for f in dl:
        print("Beginning vocabulary build: " + f)
        path_in  = os.path.join(inDir,f)
        df = cudf.DataFrame()
        with open(path_in, 'rb') as fi:
            df['tweets'] = pickle.load(fi)

        # basic preprocessing
        df = basic_clean(df)
        countvec.partial_fit(df['tweets'])
        print("End " + f)

        # count rows of data
        num_data_rows += len(df.index)
    pickle.dump(countvec, open(COUNTVECTOR_FILEPATH, 'wb'))


if vocab_build== 0:
    countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))


if transform_mean==1:
    i=0
    # compute global mean of the vocab frequencies
    vocab = len(countvec.vocabulary_)
    print("right after countvec partial fit vocab\n\n\n: ", vocab)
    M1_sum = tl.zeros(vocab)
    tot_len = 0
    for f in dl:
        i+=1
        if i % 100==0 :
            print("Beginning transform/mean: " + f)
        path_in  = os.path.join(inDir,f)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        # read in dataframe 
        df = pd.DataFrame()
        with open(path_in, 'rb') as fi:
            df['tweets'] = pickle.load(fi)
        mask = df['tweets'].str.len() > 10 
        df   = df.loc[mask]
        df   = cudf.from_pandas(df)
        # basic preprocessing
        df   = basic_clean(df)

        X_batch = tl.tensor(countvec.transform(df['tweets']).toarray())
        M1_sum += tl.sum(X_batch, axis=0)
        if i % 100==0 :
            print(X_batch.shape[1])
        tot_len += X_batch.shape[0]
        if i % 100==0 :
            print(str(tot_len))
        if save_files == 1:
            
            pickle.dump(
                X_batch, 
                open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','wb')
            )
        print("End " + f)

    M1 = M1_sum/tot_len
    print("Total length of dataset: {} rows".format(str(tot_len)))

    pickle.dump(countvec, open(COUNTVECTOR_FILEPATH, 'wb'))
    pickle.dump(M1, open(M1_FILEPATH, 'wb'))
    pickle.dump(tot_len, open(TOTAL_DATA_ROWS_FILEPATH, 'wb'))
    del M1_sum
    del X_batch 
    del df
    del mask
    gc.collect()

if vocab_build == 0:
    countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))
    M1       = pickle.load(open(M1_FILEPATH,'rb'))
    print("vocab: M1 shape: ", M1.shape)
    vocab = len(countvec.vocabulary_)
    print("vocab: vocab shape: ",vocab)

gc.collect()


pca = PCA(num_tops, alpha_0, batch_size_pca,backend)

if pca_run == 1:
    t1 = time.time()
    X_batch = None
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
    
        print("Beginning PCA: " + f)
        if X_batch is None:
            X_batch = pickle.load(
                        open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                        #open(f,'rb')
                    )
                
        else:
            temp = pickle.load(
            open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
            #open(f,'rb')
                        )
                    
            X_batch = cp.append(X_batch,temp,0)
            del temp
            gc.collect()

        
        if X_batch.shape[0] >= 55000:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            
            print("M1 shape: ", M1.shape)
            print("X batch: ", X_batch.shape)


            for j in range(0, len(X_batch)-(batch_size_pca-1), batch_size_pca):
                k = j + batch_size_pca

                # Check if remainder is undersized
                if (len(X_batch) - k) < batch_size_pca:
                    k = len(X_batch)
                
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

                
                y = tl.tensor(X_batch[j:k])
                y -= M1 # center the data
                pca.partial_fit(y)
            X_batch=None
            gc.collect()

    t2 = time.time()
    print("PCA and Centering Time: " + str(t2-t1))
 #   print(dir(pca.pca))
    pickle.dump(pca, open(PCA_FILEPATH,'wb'))
    pickle.dump(pca.projection_weights_, open(PCA_PROJ_WEIGHTS_FILEPATH,'wb'))
    pickle.dump(pca.whitening_weights_, open(PCA_WHITENING_WEIGHTS_FILEPATH,'wb'))
    del X_batch 
    del y

gc.collect()
if pca_run ==0:
    pca = pickle.load(open(PCA_FILEPATH,'rb'))

gc.collect()

if whiten == 1:
    t1 = time.time()
    x_whits = []
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        print("Beginning TLDA: " + f)
        X_batch = cp.ndarray.get(pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                    #open(f,'rb')
                )
            )
       
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        
        t1 = time.time()
        

        # Check if remainder is undersized

        
        y = tl.tensor(X_batch)
        y -= M1 # center the data
        x_whits.append(pca.transform(y))
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()

        t2 = time.time()
        print("New whiten time " + str(t2-t1))
    x_whit = tl.concatenate(x_whits, axis=0)
    print(x_whit.shape)
    pickle.dump(x_whit, open(X_WHITENED_FILEPATH,'wb'))
    t2 = time.time()
 
    print("Whiten time: " + str(t2-t1))

if whiten == 0:
    x_whit= pickle.load(open(X_WHITENED_FILEPATH,'rb'))

if stgd == 1:
    M3=None
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,cumulant = M3,gamma_shape = 1.0, smoothing = 1e-5,theta=theta_param, ortho_loss_criterion = ortho_loss_param)
    t1 = time.time()
    tlda.fit(x_whit,pca,M1,vocab,verbose=True)
    t2 = time.time()
    tlda_time =str(t2-t1)
    print("TLDA Time: " + tlda_time)

    pickle.dump(cp.asnumpy(tlda.factors_), open(TLDA_FILEPATH, 'wb'))
    outFile = open(WEIGHTS_FILEPATH, 'w')
    print(tlda.weights_, file=outFile)
    print(np.argsort(cp.asnumpy(tlda.weights_))[::-1], file=outFile)
    outFile.close()

if stgd == 0:
        tlda               = pickle.load(open(TLDA_FILEPATH, 'rb'))



n_top_words = 20
df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
df_voc.to_csv(VOCAB_FILEPATH)

for k in range(0,num_tops):
    if k ==0:
        t_n_indices   =  tlda.factors_[:,k].argsort()[:-n_top_words - 1:-1]
        top_words_LDA = countvec.vocabulary_[t_n_indices]
        top_words_df  = cudf.DataFrame({'words_'+str(k):top_words_LDA}).reset_index(drop=True)
        
    if k >=1:
        t_n_indices   =  tlda.factors_[:,k].argsort()[:-n_top_words - 1:-1]
        top_words_LDA = countvec.vocabulary_[t_n_indices]
        top_words_df['words_'+str(k)] = top_words_LDA.reset_index(drop=True)


top_words_df.to_csv(TOP_WORDS_FILEPATH)


del df_voc, countvec,top_words_LDA 

if coherence == 1:
    i=1
    for f in dl:             
            print(f)
            X_batch = cupyx.scipy.sparse.csr_matrix( pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')))
            if i == 1 :
                X= X_batch
            else: 
                X       = cupyx.scipy.sparse.vstack([X,X_batch])
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

            i +=1
    n = X.shape[0]
    tcm = X.T.dot(X)
    print(tcm.shape)
    numerator   = cupyx.scipy.sparse.triu(tcm, k=1)
    denominator = M1
    print(denominator.shape)
    score       = cp.log(((numerator.toarray()/n)+epsilon)/denominator)
    topic_coh   = []
    for k in range(0,num_tops):
        t_n_indices   = tlda.factors_[:,k].argsort()[:-n_top_words - 1:-1]
        score_tmp     = score[cp.ix_(t_n_indices,t_n_indices)]
        topic_coh.append(score_tmp.mean())
 
    u_mass = sum(topic_coh)/k
    print(u_mass)



         
    
    
    
## load in X_batch, convert to sparse CP, append, X.t*X_MAT_FILEPATH_PREFIX

# sum_{i<j} log(D(wi,wj)+1/D(wi))





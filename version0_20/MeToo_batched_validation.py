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
from pca        import PCA
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
X_MAT_FILEPATH_PREFIX = '../data/Meena_testing/x_mat/' # path to store the document-term matrix
COUNTVECTOR_FILEPATH  = '../data/Meena_testing/countvec.obj' # store the count vectorizer and the tokens
M1_FILEPATH           = '../data/Meena_testing/M1.obj' # store first moment ie the mean
PCA_FILEPATH          = '../data/Meena_testing/pca.obj' # store the results from the first PCA on M1
PCA_PROJ_WEIGHTS_FILEPATH      = '../data/Meena_testing/pca_proj_weights.obj' # Store the projectin weights from PCA
PCA_WHITENING_WEIGHTS_FILEPATH = '../data/Meena_testing/pca_whitening_weights.obj' # store the whitening weight from PCA
X_WHITENED_FILEPATH = '../data/Meena_testing/x_whit.obj' # Store the whitened data
TLDA_FILEPATH       = '../data/Meena_testing/tlda.obj' # store the TLDA object
PREPROCESS_FACTORS_METOO_FILEPATH = '../data/Meena_testing/preprocess_factors_MeToo.obj' # save pre-processed factors
POST_FACTORS_METOO_FILEPATH       = '../data/Meena_testing/postprocess_factors_MeToo.obj' # save post-process factors
TOP_WORDS_FILEPATH                = '../data/top_words.csv' # save the top words per topic
VOCAB_FILEPATH                    = '../data/vocab.csv' # save the vocab
TOTAL_DATA_ROWS_FILEPATH          = '../data/total_data_rows.obj'  # save length of data. 

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
    self.fit(data)
    vocab = vocab.append(self.vocabulary_)
    self.vocabulary_ = vocab.unique()

def tune_filesplit_size_on_IPCA_batch_size(IPCA_batchsize):
    return None


# declare the stop words 
stop_words = (stopwords.words('english'))
added_words = ["sexual","metoo","womensmarch","thread","say","will","has","by","for","hi","hey","hah","thank","watch","doe",
               "said","talk","congrats","congratulations","are","as","i", "time","year","mani","trump",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement",
               "you", "your", "yours","he","her","him","she","hers","that","whi","feel","say","gt",
               "be","with","their","they're","is","was","been","not","they","way","thi",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell",
               "good","day","work", "latest","today","becaus","peopl","via","see","old","ani",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"thing","can't","can","right","got","show",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need","us", "men", "women", "get", "woman", "man", "amp","amp&","yr","yrs"]



stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit


# set you text pre-processing params 
countvec = CountVectorizer( stop_words = stop_words, # works
                            lowercase = True, # works
                            ngram_range = (1,2), ## allow for bigrams (might want to try tri-gram)
                            # toggle these two argumets so that you have 2000 words total in the dictionary
                             max_df = 500000, # limit this to 10,000
                             min_df = 2500) ## limit this to 20 


inDir  = "../data/MeTooMonthCleaned" # input('Name of input directory? : ')


n_topic = list(5,10,20,30,40,50,60,70,80,90,100,200)
# Learning parameters
num_tops = 100 # 50 topics :(931, 93, 1258) coherence: 2277 (lr=0.00003 )
alpha_0 = 0.01
batch_size_pca  = 20000  # this will handle 2000 words + 100 topics ad infinite number of documents 
#batch_size_pca  = 5000 # for whitening
batch_size_grad = 8000 #divide data by 1,000 ## 800 = -3322.32 (6000 seecond) 4000=-3320 (1800 seconds) 8000=-3325 (1180 seconds)  Lower this to 1% of TOTAL data size
n_iter_train    = 1000
n_iter_test     = 1
learning_rate   = 0.00001 #30 topics # 0.00001 8000=-3325 (1180 seconds); 0.00002 8000=-3321 (452 seconds); 0.00003 8000=-3322 (275 seconds);  0.00004 8000=-3322 (907 seconds);
theta_param = 5.005
smoothing   = 1e-7

# Program controls
split_files = 0
vocab_build = 0
save_files  = 0
pca_run     = 0
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


# Split datafiles into smaller files (makes memory mangement easy)
print("Splitting files")

if split_files == 1:
    inDir = fop.split_files(
        inDir, 
        os.path.join(
            "../data/MeTooMonthCleaned", 
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

        #####!!!!!!!!! Read in the file as  a list and convert  to cudf (convert the pickled list to cudf dataframe)
        # read in dataframe 
        df = cudf.read_csv(path_in, names = ['tweets'])

        # basic preprocessing
        df = basic_clean(df)
        countvec.partial_fit(df['tweets'])
        print("End " + f)

        # count rows of data
        num_data_rows += len(df.index)

    # compute global mean of the vocab frequencies
    vocab = len(countvec.vocabulary_)
    print("right after countvec partial fit vocab\n\n\n: ", vocab)
    M1_sum = tl.zeros(vocab)
    tot_len = 0
    for f in dl:
        print("Beginning transform/mean: " + f)
        path_in  = os.path.join(inDir,f)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        # read in dataframe 
        df = pd.read_csv(path_in, names = ['tweets'])
        mask = df['tweets'].str.len() > 10 
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
    num_data_rows = 7976703 # delete after testing

gc.collect()


pca = PCA(num_tops, alpha_0, batch_size_pca,backend)

if pca_run == 1:
    t1 = time.time()
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
    
        print("Beginning PCA: " + f)

        X_batch = cp.ndarray.get(pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                    #open(f,'rb')
                )
            )
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

    t2 = time.time()
    print("PCA and Centering Time: " + str(t2-t1))
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
            x_whits.append(pca.transform(y))
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

 
    x_whit = tl.concatenate(x_whits, axis=0)
    print(x_whit.shape)
    pickle.dump(x_whit, open(X_WHITENED_FILEPATH,'wb'))
    t2 = time.time()
 
    print("Whiten time: " + str(t2-t1))

if whiten == 0:
    x_whit= pickle.load(open(X_WHITENED_FILEPATH,'rb'))
gc.collect()
if stgd == 1:
    M3=None
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,cumulant = M3,gamma_shape = 1.0, smoothing = 1e-6,theta=theta_param)
    t1 = time.time()
    tlda.fit(x_whit,pca,M1,vocab,verbose=True)
    t2 = time.time()
    tlda_time =str(t2-t1)
    print("TLDA Time: " + tlda_time)

    pickle.dump(tlda, open(TLDA_FILEPATH, 'wb'))

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


del df_voc, countvec,top_words_LDA, x_whit 

if coherence == 1:
    i=1
    for f in dl:             
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
    tcm = X.T.dot(X)
    print(tcm.shape)
    numerator   = cupyx.scipy.sparse.triu(tcm, k=1)
    denominator = M1*X.shape[0]
    score       = cp.log((numerator.toarray()+1)/denominator)
    topic_coh   = []
    for k in range(0,num_tops):
        t_n_indices   = tlda.factors_[:,k].argsort()[:-n_top_words - 1:-1]
        score_tmp     = score[cp.ix_(t_n_indices,t_n_indices)]
        topic_coh.append(score_tmp.sum())
 
    u_mass = sum(topic_coh)/k
    print(u_mass)
                



         
    
    
    
## load in X_batch, convert to sparse CP, append, X.t*X_MAT_FILEPATH_PREFIX

# sum_{i<j} log(D(wi,wj)+1/D(wi))





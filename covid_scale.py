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
nltk.download('stopwords')
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
from version0_99.tlda import TLDA
from version0_99.pca        import PCA
# import tensor_lda_mid as tlda_mid
# import test_util_validation
# import tensor_lda_util as tl_util
import version0_99.file_operations as fop

# class StemTokenizer(object):
#     def __init__(self):
#         self.porter = PorterStemmer()
#     def __call__(self, articles):
#         return [self.porter.stem(t) for t in word_tokenize(articles)]


# Constants
X_MAT_FILEPATH_PREFIX = '/raid/covid_scaling/data/x_mat/' # path to store the document-term matrix
COUNTVECTOR_FILEPATH  = '/raid/covid_scaling/data/countvec_1M.obj' # store the count vectorizer and the tokens
M1_FILEPATH           = '/raid/covid_scaling/data/M1.obj' # store first moment ie the mean
PCA_FILEPATH          = '/raid/covid_scaling/data/pca.obj' # store the results from the first PCA on M1
PCA_PROJ_WEIGHTS_FILEPATH      = '/raid/covid_scaling/data/pca_proj_weights.obj' # Store the projectin weights from PCA
PCA_WHITENING_WEIGHTS_FILEPATH = '/raid/covid_scaling/data/pca_whitening_weights.obj' # store the whitening weight from PCA
X_WHITENED_FILEPATH = '/raid/covid_scaling/data/x_whit.obj' # Store the whitened data
TLDA_FILEPATH       = '/raid/covid_scaling/data_covid.obj' # store the TLDA object
WEIGHTS_FILEPATH    = '/raid/covid_scaling/data/weights_tlda_covid.txt'
PREPROCESS_FACTORS_METOO_FILEPATH = '/raid/covid_scaling/data/preprocess_factors_MeToo.obj' # save pre-processed factors
POST_FACTORS_METOO_FILEPATH       = '/raid/covid_scaling/data/postprocess_factors_MeToo.obj' # save post-process factors
TOP_WORDS_FILEPATH                = '/raid/covid_scaling/data/top_words_covid.csv' # save the top words per topic
VOCAB_FILEPATH                    = '/raid/covid_scaling/data/vocab.csv' # save the vocab
TOTAL_DATA_ROWS_FILEPATH          = '/raid/covid_scaling/data/total_data_rows.obj'  # save length of data. 

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
porter = PorterStemmer()


def basic_clean(df):
    df['tweet'] = df['tweet'].astype('str')
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
added_words = ["thread","say","will","has","by","for","hi","hey","hah","thank","watch","doe","covid"
               "said","talk","congrats","congratulations","are","as","i", "time","abus","year","mani",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement","assault",
               "you", "your", "yours","he","her","him","she","hers","that","harass","whi","feel","say","gt",
               "be","with","their","they're","is","was","been","not","they","way","thi",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell",
               "good","day","work", "latest","today","becaus","peopl","via","see","timesup","old","ani",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"thing","can't","can","right","got","show",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need","us", "men", "women", "get", "man", "amp","amp&","yr","yrs",
               "shirt", "vs"]



stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit


# set you text pre-processing params 
countvec = CountVectorizer( stop_words = stop_words, # works
                            lowercase = True, # works
                            ngram_range = (1,2), ## allow for bigrams
                            # toggle these two argumets so that you have 2000 words total in the dictionary
                            #max_df = 10000, # limit this to 10,000 ## 500000 for 8M
                            max_df = 100000, #100000, # limit this to 10,000 ## 500000 for 8M
                            min_df = 2500)# 2000) ## limit this to 20 ## 2500 for 8M


inDir  = "/raid/covid_scaling/data/split_files" # MeTooMonthCleaned" # input('Name of input directory? : ')

# Learning parameters
num_tops = 20 #100 # 50 topics :(931, 93, 1258) coherence: 2277 (lr=0.00003 )
alpha_0 = 0.01
batch_size_pca  = 100000  # this will handle 2000 words + 100 topics ad infinite number of documents 
#batch_size_pca  = 5000 # for whitening
batch_size_grad = 25000 # 1% of data size - see what coherence looks like - can also try increasing  #divide data by 1,000 ## 800 = -3322.32 (6000 seecond) 4000=-3320 (1800 seconds) 8000=-3325 (1180 seconds)  Lower this to 1% of TOTAL data size
n_iter_train    = 1000
n_iter_test     = 1
# 0.0005
learning_rate   = 0.005 # increase bc increased batch size #30 topics # 0.00001 8000=-3325 (1180 seconds); 0.00002 8000=-3321 (452 seconds); 0.00003 8000=-3322 (275 seconds);  0.00004 8000=-3322 (907 seconds);
theta_param = 5.005
smoothing   = 1e-5
ortho_loss_param = 40

# Program controls
split_files = 0
vocab_build = 0
save_files  = 0
pca_run     = 0
whiten      = 0
stgd        = 1
coherence   = 1
# Other globals
num_data_rows = 0
max_data_rows = 1.2e6

#Start

print("\n\nSTART...")

# tlda = pickle.load(open(TLDA_FILEPATH, "rb"))
# pickle.dump(tlda.factors_, open("../data_factors_metoo.obj", "wb"))

# Get a list of files in the directory
dl = os.listdir(inDir)

# FIRST FILE HAS 1M Tweets
# first 11 files have 2M tweets
# first 21 files have 5M tweets, 41 split
#dl = sorted(fop.get_files_in_dir(inDir))[8:8+MONTHS]
#dl = sorted(fop.get_files_in_dir(inDir))[:41]# [:21]

# Split datafiles into smaller files (makes memory mangement easy)
print("Splitting files")

if split_files == 1:
    inDir = fop.split_files(
        inDir, 
        os.path.join(
            "data", 
            "split_files"),
        size_threshold = 100000000
    )
    dl = fop.get_files_in_dir(inDir) # we sort so that they are ordered in chronological order
    print("Done. Split files located at: {}.\n".format(inDir))
    print("Split files and their filesizes: ")
    fop.print_filesizes(inDir)
    #fop.print_num_rows_in_csvs(inDir)

# Build the vocabulary
if vocab_build == 1:
    for i, f in enumerate(dl):
        print("Beginning vocabulary build: " + f)
        path_in  = os.path.join(inDir,f)
        
      

        #####!!!!!!!!! Read in the file as  a list and convert  to cudf (convert the pickled list to cudf dataframe)
        # read in dataframe 
        df = cudf.read_csv(path_in)

        # basic preprocessing
        df = basic_clean(df)
        countvec.partial_fit(df['tweet'])
        
        
        print("End " + f)
        num_data_rows += len(df.index)
        if num_data_rows < 10000000:
            print(df.head())
        # count rows of data
        print(num_data_rows)
        print(len(df.index))
        print( len(countvec.vocabulary_))
    # compute global mean of the vocab frequencies
    pickle.dump(countvec, open(COUNTVECTOR_FILEPATH, 'wb'))
    # countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))
    vocab = len(countvec.vocabulary_)
    
    print("right after countvec partial fit vocab\n\n\n: ", vocab)
    M1_sum = tl.zeros(vocab)
    len_arr = []
    tot_len = 0
    for f in dl:
        print("Beginning transform/mean: " + f)
        path_in  = os.path.join(inDir,f)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        # read in dataframe 
        df = pd.read_csv(path_in)
        mask = df['tweet'].str.len() > 10 
        df   = df.loc[mask]
        print(df.head())
        df   = cudf.from_pandas(df)
        # basic preprocessing
        df   = basic_clean(df)

        X_batch = tl.tensor(countvec.transform(df['tweet']).toarray()) #oarray())
        M1_sum += tl.sum(X_batch, axis=0)
        print(X_batch.shape[0])
        len_arr.append(X_batch.shape[0])
        tot_len += X_batch.shape[0]
        print(str(tot_len))
        if save_files == 1:
            pickle.dump(
                (X_batch), 
                open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '_' + str(num_tops) + '.obj','wb')
            )
        print("End " + f)

    M1 = M1_sum/tot_len
    print(len_arr)
    print("Total length of dataset: {} rows".format(str(tot_len)))

    df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
    df_voc.to_csv(VOCAB_FILEPATH)

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
    num_data_rows = 132871555# 7976703 # delete after testing

gc.collect()


pca = PCA(num_tops, alpha_0, batch_size_pca,backend)

if pca_run == 1:
    t1 = time.time()
    new_pca_time=0
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
    
        print("Beginning PCA: " + f)

        # cp.ndarray.get()
        X_batch = pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '_' + str(num_tops) + '.obj','rb')
                    #open(f,'rb')
                )
            # )
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        
        print("M1 shape: ", M1.shape)
        print("X batch: ", X_batch.shape)

        t0 = time.time()
        for j in range(0, max(1, len(X_batch)-(batch_size_pca-1)), batch_size_pca):
            k = j + batch_size_pca

            # Check if remainder is undersized
            if (len(X_batch) - k) < batch_size_pca:
                k = len(X_batch)
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

            # tl.tensor()
            y = tl.tensor(X_batch[j:k])
            y -= M1 # center the data
            pca.partial_fit(y)
        t2 = time.time()
        new_pca_time += t2-t0
    t2 = time.time()
    print("NEW PCA TIME: " + str(new_pca_time))
    print("PCA and Centering Time: " + str(t2-t1))
    pickle.dump(pca, open(PCA_FILEPATH,'wb'))
    pickle.dump(pca.projection_weights_, open(PCA_PROJ_WEIGHTS_FILEPATH,'wb'))
    pickle.dump(pca.whitening_weights_, open(PCA_WHITENING_WEIGHTS_FILEPATH,'wb'))
    del X_batch 
    del y

gc.collect()
if pca_run ==0:
    pca = pickle.load(open(PCA_FILEPATH,'rb'))
idx=1
gc.collect()
if whiten == 1:
    t0 = time.time()
    x_whits = []
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        print("Beginning TLDA: " + f)
        print(str(idx) +'/'+str(726))
        idx+=1
        # cp.ndarray.get(
        X_batch = pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '_' + str(num_tops) + '.obj','rb')
                    #open(f,'rb')
                )
            # )
       
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        
        t1 = time.time()
        for j in range(0, max(1, len(X_batch)-(batch_size_pca-1)), batch_size_pca):
            k = j + batch_size_pca

            # Check if remainder is undersized
            if (len(X_batch) - k) < batch_size_pca:
                k = len(X_batch)
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            y = tl.tensor(X_batch[j:k])
            # y = X_batch[j:k]
            y -= M1 # center the data
            x_whits.append(pca.transform(y))
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        t2 = time.time()
        print("New whiten time" + str(t2-t0))
    t2 = time.time()
 
    print("Whiten time: " + str(t2-t0))
    pickle.dump(x_whits, open(X_WHITENED_FILEPATH,'wb'))



if whiten == 0:
    x_whits= pickle.load(open(X_WHITENED_FILEPATH,'rb'))
gc.collect()
if stgd == 1:
    M3=None
    tlda = TLDA(num_tops, alpha_0, n_iter_train,n_iter_test ,batch_size_grad ,learning_rate,cumulant = M3,gamma_shape = 1.0, smoothing = 1e-5,theta=theta_param, ortho_loss_criterion = ortho_loss_param)
    t1 = time.time()
    for x_whit in x_whits:
        tlda.fit(x_whit,pca,M1,vocab,verbose=False)
    t2 = time.time()
    tlda_time =str(t2-t1)
    print("TLDA Time: " + tlda_time)

    pickle.dump(cp.asnumpy(tlda.factors_), open(TLDA_FILEPATH, 'wb'))
    outFile = open(WEIGHTS_FILEPATH, 'w')
    print(tlda.weights_, file=outFile)
    print(np.argsort(cp.asnumpy(tlda.weights_))[::-1], file=outFile)
    outFile.close()
'''
if stgd == 0:
        tlda               = pickle.load(open(TLDA_FILEPATH, 'rb'))



n_top_words = 20

df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
df_voc.to_csv(VOCAB_FILEPATH)
print("tlda factors shape: " + str(tlda.factors_.shape))
print("tlda theta: " + str(tlda.theta))

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


del df_voc, countvec,top_words_LDA, x_whit''' 
epsilon = 1e-12
if vocab_build == 0:
    M1       = pickle.load(open(M1_FILEPATH,'rb'))

if coherence == 1:
    n_top_words = 20

    i=1
    for f in dl:             
            print(f)
            X_batch = cupyx.scipy.sparse.csr_matrix( pickle.load(
                    open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '_' + str(num_tops) + '.obj','rb')))
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
    #M1 = cp.mean(X, axis=0)
    tcm = X.T.dot(X)
    print(tcm.shape)
    numerator   = cupyx.scipy.sparse.triu(tcm, k=1)
    #cupyx.scipy.sparse.triu(tcm, k=1)
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





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

#Insert Plotly
import pandas as pd
import time
import pickle

# Import utility functions from other files
from version0_99.tlda_wrapper import TLDA
import version0_99.file_operations as fop



# Constants

X_MAT_FILEPATH_PREFIX = '/raid/debanks/MeToo/data/x_mat/' # path to store the document-term matrix
COUNTVECTOR_FILEPATH  = '/raid/debanks/MeToo/data/countvec_1M.obj' # store the count vectorizer and the tokens
TLDA_FILEPATH       = '/raid/debanks/MeToo/data/tlda_metoo_comparison.obj' # store the TLDA object
VOCAB_FILEPATH                    = '/raid/debanks/MeToo/data/vocab.csv' # save the vocab
TOPIC_FILEPATH_PREFIX   = '/raid/debanks/MeToo/data/predicted_topics/'
DOCUMENT_TOPIC_FILEPATH = '/raid/debanks/MeToo/data/dtm.csv'
DOCUMENT_TOPIC_FILEPATH_TOT = '/raid/debanks/MeToo/data/dtm_df.csv'
RAW_DATA_PREFIX = '/raid/debanks/MeToo/data/MeTooMonth/'
OUT_ID_DATA_PREFIX = '/raid/debanks/MeToo/data/ids/' 
TOP_WORDS_FILEPATH ='/raid/debanks/MeToo/data/top_words.csv'

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
added_words = ["thread","say","will","has","by","for","hi","hey","hah","thank","metoo","watch","sexual","doe",
               "said","talk","congrats","congratulations","are","as","i", "time","abus","year","mani","trump",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement","assault",
               "you", "your", "yours","he","her","him","she","hers","that","harass","whi","feel","say","gt",
               "be","with","their","they're","is","was","been","not","they","womensmarch","way","thi",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell",
               "good","day","work", "latest","today","becaus","peopl","via","see","timesup","old","ani",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"thing","can't","can","right","got","show",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need","us", "men", "women", "get", "woman", "man", "amp","amp&","yr","yrs", "heforsh", "daca",
               "shirt", "resist", "vs"]



stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit


# set you text pre-processing params 
countvec = CountVectorizer( stop_words = stop_words, # works
                            lowercase = True, # works
                            ngram_range = (1,2), ## allow for bigrams
                            # toggle these two argumets so that you have 2000 words total in the dictionary
                            #max_df = 10000, # limit this to 10,000 ## 500000 for 8M
                            max_df = 0.5, #100000, # limit this to 10,000 ## 500000 for 8M
                            min_df = 0.005)# 2000) ## limit this to 20 ## 2500 for 8M


inDir  = "/raid/debanks/MeToo/data/MeTooMonthCleaned" # MeTooMonthCleaned" # input('Name of input directory? : ')

# Learning parameters
num_tops = 100 #100 # 50 topics :(931, 93, 1258) coherence: 2277 (lr=0.00003 )
alpha_0 = 0.01
batch_size_pca  = 50000  # this will handle 2000 words + 100 topics ad infinite number of documents 
#batch_size_pca  = 5000 # for whitening
batch_size_grad = 750 # 1% of data size - see what coherence looks like - can also try increasing  #divide data by 1,000 ## 800 = -3322.32 (6000 seecond) 4000=-3320 (1800 seconds) 8000=-3325 (1180 seconds)  Lower this to 1% of TOTAL data size
n_iter_train    = 20
n_iter_test     = 10
# 0.0005
learning_rate   = 0.0004 # increase bc increased batch size #30 topics # 0.00001 8000=-3325 (1180 seconds); 0.00002 8000=-3321 (452 seconds); 0.00003 8000=-3322 (275 seconds);  0.00004 8000=-3322 (907 seconds);
theta_param = 5.005
smoothing   = 1e-7
ortho_loss_param = 1000

# Program controls
split_files    = 0
vocab_build    = 0
save_files     = 0
stgd           = 1
recover_top_words = 1
transform_data    = 0
create_meta_df    = 0
coherence         = 1

# Other globals
num_data_rows = 0
max_data_rows = 1.2e6

#Start

print("\n\nSTART...")



dl = sorted(fop.get_files_in_dir(inDir))



if split_files == 1:
    print("Splitting files")
    inDir = fop.split_files(
        inDir, 
        os.path.join(
            "/raid/debanks/MeToo/data", 
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
        path_in      = os.path.join(inDir,f)
        path_in_raw  = os.path.join(RAW_DATA_PREFIX,f)
        #####!!!!!!!!! Read in the file as  a list and convert  to cudf (convert the pickled list to cudf dataframe)
        # read in dataframe 
        df = cudf.read_csv(path_in, names = ['tweets'])
        # basic preprocessing
        df = basic_clean(df)
        countvec.partial_fit(df['tweets'])
        print("End " + f)

        # count rows of data
        num_data_rows += len(df.index)
        print(num_data_rows)
        print(len(df.index))
    # compute global mean of the vocab frequencies
    pickle.dump(countvec, open(COUNTVECTOR_FILEPATH, 'wb'))
    # countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))
    vocab = len(countvec.vocabulary_)
    
    print("right after countvec partial fit vocab\n\n\n: ", vocab)
    # M1_sum = tl.zeros(vocab)
    len_arr = []
    tot_len = 0
    for f in dl:
        print("Beginning transform/mean: " + f)
        path_in  = os.path.join(inDir,f)
        path_out_ids = os.path.join(OUT_ID_DATA_PREFIX,f)
        path_in_raw  = os.path.join(RAW_DATA_PREFIX,f)

        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        # read in dataframe 
        df = pd.read_csv(path_in, names = ['tweets'])
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        mask = df['tweets'].str.len() > 10 
        df   = df.loc[mask]
        df   = cudf.from_pandas(df)
        # basic preprocessing
        df   = basic_clean(df)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        gc.collect()

        if save_files == 1:
            X_batch = tl.tensor(countvec.transform(df['tweets']).toarray()) 
            print(X_batch.shape[0])
            pickle.dump(
                (X_batch), 
                open(X_MAT_FILEPATH_PREFIX + Path(f).stem + '_' + str(num_tops) + '.obj','wb')
            )
            del X_batch 
        print("End " + f)

    df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
    df_voc.to_csv(VOCAB_FILEPATH)

    pickle.dump(countvec, open(COUNTVECTOR_FILEPATH, 'wb'))
    # del M1_sum

    del df
    del mask
    gc.collect()


if vocab_build == 0:
    countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))
    # M1       = pickle.load(open(M1_FILEPATH,'rb'))
    # print("vocab: M1 shape: ", M1.shape)
    vocab = len(countvec.vocabulary_)
    print("vocab: vocab shape: ",vocab)
    # num_data_rows = 11192442# 7976703 # delete after testing

gc.collect()


tlda = TLDA(
    num_tops, alpha_0, n_iter_train, n_iter_test,learning_rate, 
    pca_batch_size = batch_size_pca, third_order_cumulant_batch = batch_size_grad, 
    gamma_shape = 1.0, smoothing = 1e-5, theta=theta_param, ortho_loss_criterion = ortho_loss_param
)

gc.collect()
if stgd == 1:
    t1 = time.time()
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        print("Beginning TLDA: " + f)
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
        
        t3 = time.time()
        for j in range(0, max(1, len(X_batch)-(batch_size_grad-1)), batch_size_grad):
            k = j + batch_size_grad

            # Check if remainder is undersized
            if (len(X_batch) - k) < batch_size_grad:
                k = len(X_batch)
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            y = tl.tensor(X_batch[j:k])
            
            tlda.partial_fit_online(y)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        t4 = time.time()
        print("New fit time" + str(t4-t3))
    t2 = time.time()
 
    print("Fit time: " + str(t2-t1))
    pickle.dump(tlda, open(TLDA_FILEPATH, 'wb'))


# Document Processing

if stgd == 0:
    tlda = pickle.load(open(TLDA_FILEPATH,'rb'))
    tlda.unwhitened_factors_= tlda._unwhiten_factors()
    # M1       = pickle.load(open(M1_FILEPATH,'rb'))
    # print("vocab: M1 shape: ", M1.shape)

    print("Load TLDA")
    # num_data_rows = 11192442# 7976703 # delete after testing


if recover_top_words == 1:
    n_top_words = 20

    print(tlda.unwhitened_factors_.shape)    


    for k in range(0,num_tops): 
        if k ==0:
            t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
            top_words_LDA = countvec.vocabulary_[t_n_indices]
            top_words_df  = cudf.DataFrame({'words_'+str(k):top_words_LDA}).reset_index(drop=True)
            
        if k >=1:
            t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
            top_words_LDA = countvec.vocabulary_[t_n_indices]
            top_words_df['words_'+str(k)] = top_words_LDA.reset_index(drop=True)


    top_words_df.to_csv(TOP_WORDS_FILEPATH)



if transform_data == 1:
    print("Unwhiten Factors")
    tlda.unwhitened_factors_= tlda._unwhiten_factors()
    t1  = time.time()
    dtm = None
    for f in dl:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        print("Beginning Document Fitting: " + f)
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
        batch_size_grad = 100000
        t3 = time.time()
        for j in range(0, max(1, len(X_batch)-(batch_size_grad-1)), batch_size_grad):
            k = j + batch_size_grad

            # Check if remainder is undersized
            if (len(X_batch) - k) < batch_size_grad:
                k = len(X_batch)
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            y = tl.tensor(X_batch[j:k])
            
            if dtm is not None:
                dtm = tl.concatenate((dtm,tlda.transform(y)),axis=0)
            else:
                dtm = tlda.transform(y)

            print(dtm.shape)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        t4 = time.time()
        print("New fit time" + str(t4-t3))
        del X_batch

    t2 = time.time()

    print("Fit time: " + str(t2-t1))  

    pickle.dump(cp.asnumpy(dtm), open(DOCUMENT_TOPIC_FILEPATH, 'wb'))



epsilon = 1e-12
if vocab_build == 0:
    M1       = pickle.load(open(TLDA_FILEPATH, 'rb')).mean


if create_meta_df==1:
    print("Create MetaData")
    for f in dl:
        print("Beginning MetaData Creation: " + f)

        gc.collect()
        df = pd.read_csv(os.path.join(RAW_DATA_PREFIX,f),lineterminator='\n')
        mask = df['tweets'].str.len() > 10 
        df   = df.loc[mask]
        df   = cudf.from_pandas(df)
        df   = basic_clean(df)

        gc.collect()
        df.to_csv(OUT_ID_DATA_PREFIX+f)







if coherence == 0:
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
    tcm = X.T.dot(X)
    print(tcm.shape)
    numerator   = cupyx.scipy.sparse.triu(tcm, k=1)
    denominator = M1
    print(denominator.shape)
    score       = cp.log(((numerator.toarray()/n)+epsilon)/denominator)
    topic_coh   = []
    for k in range(0,num_tops):
        t_n_indices   = tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
        score_tmp     = score[cp.ix_(t_n_indices,t_n_indices)]
        topic_coh.append(score_tmp.mean())
 
    u_mass = sum(topic_coh)/k
    print(u_mass)
                

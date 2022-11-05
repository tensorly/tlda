from platform import win32_edition
import numpy as np
import cupy as cp
import scipy
import os
from os.path import exists, isfile, join
from pathlib import Path
import sys
import shutil
import gc
import math
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import json


# Import stopwords
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.util import everygrams

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


ROOT_DIR        = "/raid/debanks/covid_scaling/data/"
INDIR           = "split_files/"
RAW_DATA_PREFIX = "split_files/"

# Constants
# Output Relative paths -- do not change
X_MAT_FILEPATH_PREFIX = "x_mat/"
X_FILEPATH = "X_full.obj"
X_DF_FILEPATH = "X_df.obj"
X_LST_FILEPATH = "X_lst.obj"
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"
TLDA_FILEPATH = "tlda.obj"
VOCAB_FILEPATH = "vocab.csv"
TOPIC_FILEPATH_PREFIX   = 'predicted_topics/'
DOCUMENT_TOPIC_FILEPATH = 'dtm.csv'
COHERENCE_FILEPATH = 'coherence.obj'
DOCUMENT_TOPIC_FILEPATH_TOT = 'dtm_df.csv'
OUT_ID_DATA_PREFIX = 'ids/' 
TOP_WORDS_FILEPATH ='top_words.csv'


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
added_words = ["thread","say","will","has","by","for","hi","hey","hah","thank","watch","doe",
               "said","talk","congrats","congratulations","are","as","i", "time","abus","year","mani",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement","assault",
               "you", "your", "yours","he","her","him","she","hers","that","harass","whi","feel","say","gt",
               "be","with","their","they're","is","was","been","not","they","way","thi","rt","i","we","and","to","for","do","go",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell",
               "good","day","work", "latest","today","becaus","peopl","via","see","old","ani",
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


 # MeTooMonthCleaned" # input('Name of input directory? : ')

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

#

# declare the stop words 
stop_words = (stopwords.words('english'))
added_words = ["thread","say","will","has","by","for","hi","hey","hah","thank", "watch",
               "said","talk","congrats","congratulations","are","as", "time","year","mani","trump",
                "use","look","that","whi","feel","say","gt","to","do",
               "be","with","their","they're","is","was","been","not","they","way","thi",
               "it","have",  "one","think",   "thing"    ,"bring","put","well","take","exactli","tell",
               "good","day","work", "latest","today","becaus","peopl","via","see","old","ani",
               "call", "wouldnt","wow", "learned","hi"   , "things" ,"thing","can't","can","right","got","show",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need", "get", "amp","amp&","yr","yrs", "@", "#", "a", "b", "c", "d", "e", "f", "g", "h", "i",
               "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# set stop words and countvectorizer method
stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit

# define function with no preprocessing
def custom_preprocessor(doc):
    return doc


def fit_topics(num_tops, curr_dir, first_run = False, alpha_0 = 0.01, learning_rate = 0.0004, theta_param = 5.005, ortho_loss_param = 1000):
    # set you text pre-processing params 
    countvec = CountVectorizer( stop_words = stop_words, # works
                                lowercase = True, # works
                                ngram_range = (1,2), ## allow for bigrams
                                preprocessor = custom_preprocessor,
                                # toggle these two argumets so that you have 2000 words total in the dictionary
                                #max_df = 10000, # limit this to 10,000 ## 500000 for 8M
                                max_df = 0.5, #100000, # limit this to 10,000 ## 500000 for 8M
                                min_df = 0.005)# 2000) ## limit this to 20 ## 2500 for 8M
    
    # make final directories for outputs
    save_dir = os.path.join(ROOT_DIR, curr_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    exp_save_dir = os.path.join(save_dir, "num_tops_" + str(num_tops) + "_alpha0_" + str(alpha_0) + "_learning_rate_" + str(learning_rate) + "_theta_" + str(theta_param) + "_orthogonality_" + str(ortho_loss_param) + "/")
    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir)

    # inDir  = "/raid/debanks/MeToo/data/MeTooMonthCleaned" # MeTooMonthCleaned" # input('Name of input directory? : ')

    # PARAMS SET IN INPUTS
    # num_tops = 100 ## Try a range 10:100 
    # alpha_0 = 0.01  #0.01 to 0.1
    # learning_rate   = 0.0004 # increase bc increased batch size #30 topics # 0.00001 8000=-3325 (1180 seconds); 0.00002 8000=-3321 (452 seconds); 0.00003 8000=-3322 (275 seconds);  0.00004 8000=-3322 (907 seconds);
    # theta_param = 5.005
    # ortho_loss_param = 1000

    # DEFAULT PARAMS
    batch_size_pca  = 25000  # this will handle 2000 words + 100 topics ad infinite number of documents 
    batch_size_grad = 750 # 1% of data size - see what coherence looks like - can also try increasing  #divide data by 1,000 ## 800 = -3322.32 (6000 seecond) 4000=-3320 (1800 seconds) 8000=-3325 (1180 seconds)  Lower this to 1% of TOTAL data size
    smoothing   = 1e-5
    n_iter_train = 20
    n_iter_test = 10
    
    #SET SEED
    seed = 57

    # Program controls
    vocab_build    = first_run
    save_files     = first_run
    stgd           = 1
    recover_top_words = 1
    transform_data    = 1
    create_meta_df    = 0
    coherence         = 1

    # Other globals
    num_data_rows = 0
    # max_data_rows = 1.2e6

    #Start

    print("\n\nSTART...")

    """
    We want 
    1. Do gridsearch over topics, alpha, (LDA parameters, learning rate, theta,ortho loss param ) 
    a. Gensim coherence measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi', "perplexity" 
    danny will figure out perplexity(scatter plot of perplexity vs. Coherence)
    b. Report Word Clouds
    c. Document-topics inference
    d. Top 5 tweets from each topic
    2. For the optimal topic:
    a. Time series trends in probability of key topics (with all topics in appendix, danny will 
                                                        produce from 1.c output in R)


    To dos:
    - wrap pipeline in a function that takes in a list of params to iterate over 
    - write script to output lists of params for the terminal command 
    - Account for breaks due memory constraints being hit/others snags
    - add code for computing Gensim coherence measures (Danny)
    - add code for outputting coherence, time, parameters, (file names for outputs?) in JSON (Sara)
    - add code for outputting document/topic inference, top 5 tweets, wordclouds (Danny)

    """
    inDir = os.path.join(ROOT_DIR, INDIR)

    dl = sorted(fop.get_files_in_dir(inDir))

    # Build the vocabulary
    if vocab_build == 1:
        for i, f in enumerate(dl):
            print("Beginning vocabulary build: " + f)
            path_in      = os.path.join(inDir,f)
            # path_in_raw  = os.path.join(os.path.join(ROOT_DIR, RAW_DATA_PREFIX),f)
            #####!!!!!!!!! Read in the file as  a list and convert  to cudf (convert the pickled list to cudf dataframe)
            # read in dataframe 
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            # read in dataframe 
            df = pd.read_csv(path_in, names = ['date','id_str','tweet','user_id','screen_name','verified','retweet_count','favorite_count','location'],usecols=['tweet'])
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            mask = df['tweet'].str.len() > 10 
            df   = df.loc[mask]
            df   = cudf.from_pandas(df)
            # basic preprocessing
            df   = basic_clean(df)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            gc.collect()
            
            countvec.partial_fit(df['tweet'])
            print("End " + f)

            # count rows of data
            num_data_rows += len(df.index)
            print(num_data_rows)
            print(len(df.index))
        # compute global mean of the vocab frequencies
        pickle.dump(countvec, open(os.path.join(save_dir, COUNTVECTOR_FILEPATH), 'wb'))
        # countvec = pickle.load(open(COUNTVECTOR_FILEPATH,'rb'))
        vocab = len(countvec.vocabulary_)
        
        print("right after countvec partial fit vocab\n\n\n: ", vocab)
        x_mat_dir = os.path.join(save_dir, X_MAT_FILEPATH_PREFIX)
        if not os.path.exists(x_mat_dir):
            os.makedirs(x_mat_dir)
        corpus_dir = os.path.join(save_dir, CORPUS_FILEPATH_PREFIX)
        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)
        # M1_sum = tl.zeros(vocab)
        len_arr = []
        tot_len = 0
        for f in dl:
            print("Beginning transform/mean: " + f)
            path_in  = os.path.join(inDir,f)
            # path_out_ids = os.path.join(OUT_ID_DATA_PREFIX,f)
            # path_in_raw  = os.path.join(RAW_DATA_PREFIX,f)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            # read in dataframe 
            df = pd.read_csv(path_in, names = ['date','id_str','tweet','user_id','screen_name','verified','retweet_count','favorite_count','location'],usecols=['tweet'])
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            mask = df['tweet'].str.len() > 10 
            df   = df.loc[mask]
            df   = cudf.from_pandas(df)
            # basic preprocessing
            df   = basic_clean(df)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            gc.collect()

            if save_files == 1:
                corpus = countvec.transform(df['tweet'])
                X_batch = tl.tensor(corpus.toarray()) 
                print(X_batch.shape[0])
                pickle.dump(
                    (X_batch), 
                    open(x_mat_dir + Path(f).stem + '.obj','wb')
                )
                pickle.dump(
                    (corpus), 
                    open(corpus_dir + Path(f).stem + '.obj','wb')
                )
                del X_batch 
                del corpus
            print("End " + f)
            del df
            del mask

        df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
        df_voc.to_csv(save_dir + "/" + VOCAB_FILEPATH)

        # pickle.dump(countvec, open(save_dir + COUNTVECTOR_FILEPATH, 'wb'))
        # del M1_sum

        gc.collect()


    if vocab_build == 0:
        countvec = pickle.load(open(save_dir + "/" + COUNTVECTOR_FILEPATH,'rb'))
        # M1       = pickle.load(open(M1_FILEPATH,'rb'))
        # print("vocab: M1 shape: ", M1.shape)
        vocab = len(countvec.vocabulary_)
        print("vocab: vocab shape: ",vocab)
        # num_data_rows = 11192442# 7976703 # delete after testing

    gc.collect()


    tlda = TLDA(
        num_tops, alpha_0, n_iter_train, n_iter_test,learning_rate, 
        pca_batch_size = batch_size_pca, third_order_cumulant_batch = batch_size_grad, 
        gamma_shape = 1.0, smoothing = smoothing, theta=theta_param, ortho_loss_criterion = ortho_loss_param, random_seed = seed
    )

    tot_tlda_time = 0.0
    gc.collect()
    if stgd == 1:
        t1 = time.time()
        for f in dl:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            print("Beginfning TLDA: " + f)
            # cp.ndarray.get(
            X_batch = pickle.load(
                        open(save_dir + X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                        #open(f,'rb')
                    )
                # )
        
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            gc.collect()
            
            t3 = time.time()
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
                
                tlda.partial_fit_online(y)

                del y
                gc.collect()
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

            t4 = time.time()
            print("New fit time" + str(t4-t3))
            tot_tlda_time += t4-t3
            
            del X_batch
            gc.collect()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        t2 = time.time()
    
        print("Fit time: " + str(t2-t1))
        print("total TLDA time: " + str(tot_tlda_time))
        pickle.dump(tlda, open(exp_save_dir + TLDA_FILEPATH, 'wb'))


    # Document Processing

    if stgd == 0:
        tlda = pickle.load(open(exp_save_dir + TLDA_FILEPATH,'rb'))
        tlda.unwhitened_factors_= tlda._unwhiten_factors()
        # M1       = pickle.load(open(M1_FILEPATH,'rb'))
        # print("vocab: M1 shape: ", M1.shape)

        print("Load TLDA")
        # num_data_rows = 11192442# 7976703 # delete after testing


    if recover_top_words == 1:
        n_top_words = 100
        
        tlda.unwhitened_factors_ = tlda._unwhiten_factors()
        print(tlda.unwhitened_factors_)    


        for k in range(0,num_tops): 
            if k ==0:
                t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
                top_words_LDA = countvec.vocabulary_[t_n_indices]
                top_words_df  = cudf.DataFrame({'words_'+str(k):top_words_LDA}).reset_index(drop=True)
                
            if k >=1:
                t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
                top_words_LDA = countvec.vocabulary_[t_n_indices]
                top_words_df['words_'+str(k)] = top_words_LDA.reset_index(drop=True)


        top_words_df.to_csv(exp_save_dir + TOP_WORDS_FILEPATH)
        del top_words_df

    gc.collect()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()            
    pinned_mempool = cp.get_default_pinned_memory_pool()
    pinned_mempool.free_all_blocks()


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
                        open(save_dir + X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                        #open(f,'rb')
                    )
                # )
        
            
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            batch_size_grad = 50000
            t3 = time.time()
            
#             if dtm is not None:
#                 dtm = np.concatenate((dtm,cp.asnumpy(tlda.transform(X_batch))),axis=0) # tl.concatenate, no as numpy
#             else:
#                 dtm = cp.asnumpy(tlda.transform(X_batch)) # take out cp as numpy
            
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
                    dtm = np.concatenate((dtm,cp.asnumpy(tlda.transform(y))),axis=0) # tl.concatenate, no as numpy
                else:
                    dtm = cp.asnumpy(tlda.transform(y)) # take out cp as numpy

                print(dtm.shape)
                del y

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

            t4 = time.time()
            print("New fit time" + str(t4-t3))
            del X_batch
            
            gc.collect()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        t2 = time.time()

        print("Fit time: " + str(t2-t1))  

        pickle.dump(dtm, open(exp_save_dir + DOCUMENT_TOPIC_FILEPATH, 'wb')) # cp.asnumpy(dtm)
        del dtm
        gc.collect()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()


    epsilon = 1e-12
    if vocab_build == 0:
        M1       = pickle.load(open(exp_save_dir + TLDA_FILEPATH, 'rb')).mean


    if create_meta_df==1:
        print("Create MetaData")
        meta_dir = os.path.join(ROOT_DIR, OUT_ID_DATA_PREFIX)
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
            
        for f in dl:
            print("Beginning MetaData Creation: " + f)

            gc.collect()
            df = pd.read_csv(os.path.join(ROOT_DIR + RAW_DATA_PREFIX,f),lineterminator='\n')
            mask = df['tweet'].str.len() > 10 
            df   = df.loc[mask]
            df   = cudf.from_pandas(df)
            df   = basic_clean(df)

            gc.collect()
            
            df.to_csv(meta_dir +f)







    if coherence == 1:
        n_top_words = 20

        i=1
        for f in dl:             
            print(f)
            if not os.path.exists(save_dir + X_FILEPATH):
                X_batch = cupyx.scipy.sparse.csr_matrix( pickle.load(
                        open(save_dir + X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')))
                if i == 1 :
                    X= X_batch
                else: 
                    X       = cupyx.scipy.sparse.vstack([X,X_batch])

                    del X_batch
                    gc.collect()
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()            
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    pinned_mempool.free_all_blocks()
                
            gc.collect()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

            if not os.path.exists(save_dir + X_LST_FILEPATH):
                path_in      = os.path.join(inDir,f)
                df = pd.read_csv(path_in, names = ['tweet'])
                
                mask = df['tweet'].str.len() > 10 
                df   = df.loc[mask]
                df   = cudf.from_pandas(df)
                df = basic_clean(df)
                
                docs = countvec._preprocess(df['tweet'])
                tokenized_df = countvec._create_tokenized_df(docs)
                all_tokens = (tokenized_df.groupby('doc_id')['token'].agg(list)).reset_index(name='tokens')
                
                del tokenized_df
                del df
                gc.collect()
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
                
                curr_x_lst = all_tokens["tokens"].to_pandas().tolist()

                if i == 1:
                    X_lst = curr_x_lst
                else:
                    X_lst.extend(curr_x_lst)
                
                del all_tokens
                gc.collect()
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()            
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

#                 if i == 1:
#                     X_df = df
#                 else:
#                     X_df = pd.concat([X_df, df], ignore_index = True)
                
#                 del df
#                 pickle.dump(X_df, open(save_dir + X_DF_FILEPATH, "wb"))
#             else:
#                 X_df = pickle.load(open(save_dir + X_DF_FILEPATH, "rb"))
            
            i +=1
        
        if not os.path.exists(save_dir + X_LST_FILEPATH):
            pickle.dump(X_lst, open(save_dir + X_LST_FILEPATH, "wb"))
        else:
            X_lst = pickle.load(open(save_dir + X_LST_FILEPATH, "rb"))
        
        if not os.path.exists(save_dir + X_FILEPATH):
            pickle.dump(X, open(save_dir + X_FILEPATH, "wb"))
        else:
            X = pickle.load(open(save_dir + X_FILEPATH, "rb"))
            
            
        n = X.shape[0]
        
#         if not os.path.exists(save_dir + X_LST_FILEPATH):
#             new_tweets = X_df["tweets"].str.split()
#             X_lst = new_tweets.apply(lambda x: [' '.join(ng) for ng in everygrams(x, 1, 2)]).tolist()
#             pickle.dump(X_lst, open(save_dir + X_LST_FILEPATH, "wb"))
            
#             del new_tweets
#         else:
#             X_lst = pickle.load(open(save_dir + X_LST_FILEPATH, "rb"))
        
        
#         del X_df
#         tcm = X.T.dot(X)
#         print(tcm.shape)
#         numerator   = cupyx.scipy.sparse.triu(tcm, k=1)
#         denominator = M1
#         print(denominator.shape)
#         score       = cp.log(((numerator.toarray()/n)+epsilon)/denominator)
#         topic_coh   = []
#         for k in range(0,num_tops):
#             t_n_indices   = tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
#             score_tmp     = score[cp.ix_(t_n_indices,t_n_indices)]
#             topic_coh.append(score_tmp.mean())
    
#         u_mass = sum(topic_coh)/k
#         print(u_mass)

        ## Initialize 
        # Recover Topics
        gc.collect()
        n_top_words = 20

        print(tlda.unwhitened_factors_)    
        topics = []

        for k in range(0,num_tops): 
            if k ==0:
                t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
                topics.append(countvec.vocabulary_[t_n_indices].to_pandas().tolist())
                
            if k >=1:
                t_n_indices   =  tlda.unwhitened_factors_[:,k].argsort()[:-n_top_words - 1:-1]
                topics.append(countvec.vocabulary_[t_n_indices].to_pandas().tolist())

        del tlda
        gc.collect()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        
        # dictionary: dict mapping from words to indices
        # convert sparse to gensim corpus
        if not os.path.exists(save_dir + GENSIM_CORPUS_FILEPATH):
            common_corpus = gensim.matutils.Sparse2Corpus(X.get(), documents_columns=False)
            pickle.dump(common_corpus, open(save_dir + GENSIM_CORPUS_FILEPATH, "wb"))
        else:
            common_corpus = pickle.load(open(save_dir + GENSIM_CORPUS_FILEPATH, "rb"))
        
        #convert countvec to gensim dictionary obj
        common_dictionary = {}
        for (val, key) in enumerate(countvec.vocabulary_.to_pandas().tolist()):
            common_dictionary[val] = key
        dct = Dictionary()
        _ = dct.merge_with(common_dictionary)
        
        del countvec
        del X
        gc.collect()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        
        coherence ={}
        for cc in ['c_v', 'c_uci', 'c_npmi','u_mass']:
            t1 = time.time()
            cm = CoherenceModel(topics=topics, texts=X_lst, corpus=common_corpus, dictionary=dct, coherence=cc)
            coherence[cc] = cm.get_coherence() 
            print(coherence)
            t2 = time.time()
            del cm
            gc.collect()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            print(cc + " Coherence time: " + str(t2-t1))
            
        pickle.dump(coherence, open(exp_save_dir + COHERENCE_FILEPATH, 'wb'))
        
        del topics
        del X_lst
        del common_corpus
        del dct
        gc.collect()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        

    output_dict = {
        "coherence": coherence,
        "tlda_runtime": tot_tlda_time,
        "parameters": {
            "num_tops": num_tops,
            "alpha_0": alpha_0,
            "learning_rate": learning_rate,
            "theta": theta_param,
            "ortho_loss_param": ortho_loss_param,
            "batch_size_pca": batch_size_pca,
            "batch_size_grad": batch_size_grad,
            "smoothing": smoothing,
            "n_iter_train": 1,
            "n_iter_test": 10
        }
    }

    out_str = save_dir + "num_tops_" + str(num_tops) + "alpha0_" + str(alpha_0) + "_learning_rate_" + str(learning_rate) + "_theta_" + str(theta_param) + "_orthogonality_" + str(ortho_loss_param) + ".json"
    with open(out_str, "w") as outfile:
        json.dump(output_dict, outfile)
        
    
    gc.collect()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()            
    pinned_mempool = cp.get_default_pinned_memory_pool()
    pinned_mempool.free_all_blocks()
    

def main():
    curr_dir = "covid_experiment/"
    first_run = True
#    first_run = False

    num_tops_lst = [int(sys.argv[1])] #[20,40,60,80,100]
    alpha_0_lst  = [0.01]
    lr_lst = [0.0001]
    theta_lst = [5.005]
    ortho_lst = [1000]

    combined_lst = [[i, j, k, l, m] for i in num_tops_lst for j in alpha_0_lst for k in lr_lst for l in theta_lst for m in ortho_lst]

    for x in combined_lst:
        fit_topics(
            x[0], curr_dir, 
            first_run = first_run, 
            alpha_0 = x[1], 
            learning_rate = x[2], 
            theta_param = x[3], 
            ortho_loss_param = x[4]
        )
        first_run = False
        gc.collect()
    



if __name__ == "__main__":
    main()
                    

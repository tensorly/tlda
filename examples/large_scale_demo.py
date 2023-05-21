# basic imports
import numpy as np
import os
from pathlib import Path
import gc
import pandas as pd
import time
import pickle


# Import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Cuda imports
import cupy as cp
import cudf
from cudf import Series
from cuml.feature_extraction.text import CountVectorizer
from cuml.preprocessing.text.stem import PorterStemmer

# Import TensorLy
import tensorly as tl

# Import utility functions from other files
from .tlda.tlda_wrapper import TLDA
from .tlda.file_operations import get_files_in_dir

# Root Filepath -- can modify
ROOT_DIR = "/Users/skangaslahti/tlda/data"

# Data Relative Paths -- can modify
INDIR = "MeTooMonthCleaned/"

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
EXISTING_VOCAB_FILEPATH = "vocab.obj"
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


# declare the stop words
# potentially add extra stop words depending on the application dataset
stop_words = (stopwords.words('english'))
added_words = []

# set stop words and countvectorizer method
stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit

# define function with no preprocessing
def custom_preprocessor(doc):
    return doc


def fit_topics(num_tops, curr_dir, alpha_0 = 0.01, learning_rate = 0.0004, theta_param = 5.005, ortho_loss_param = 1000, smoothing = 1e-5, initialize_first_docs = False, n_eigenvec = None): 
    
    # make final directories for outputs
    save_dir = os.path.join(ROOT_DIR, curr_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize RAPIDS CountVectorizer
    countvec = CountVectorizer( stop_words = stop_words, 
                                lowercase = True,
                                ngram_range = (1, 2),
                                preprocessor = custom_preprocessor,
                                max_df = 0.5,
                                min_df = 0.00125)
    
    # set directory for saving CountVectorizer and TLDA
    eigenvec_str = "_n_eigenvec_" + (str(n_eigenvec) if n_eigenvec is not None else "None")
    exp_save_dir = os.path.join(save_dir, "num_tops_" + str(num_tops) + "_alpha0_" + str(alpha_0) + "_learning_rate_" + str(learning_rate) + "_theta_" + str(theta_param) + "_orthogonality_" + str(ortho_loss_param) + "_initialize_first_docs_" + str(initialize_first_docs) + eigenvec_str + "/")
    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir)

    # DEFAULT PARAMS -- Grid search according to dataset
    batch_size_pca  = 100000  
    batch_size_grad = 80000
    n_iter_train = 200
    n_iter_test = 10
    
    #SET SEED
    seed = 57

    # Program controls -- decide which portions to run
    if os.path.exists(save_dir + "/" + COUNTVECTOR_FILEPATH):
        first_run = 1   
    vocab_build    = first_run
    save_files     = first_run
    stgd           = 1
    recover_top_words = 1

    # Start
    print("\n\nSTART...")

    # Set files to read
    inDir = os.path.join(ROOT_DIR, INDIR)
    dl = sorted(get_files_in_dir(inDir))
    
    # Build the vocabulary
    if vocab_build == 1:
        if not os.path.exists(save_dir + "/" + EXISTING_VOCAB_FILEPATH):
            for i, f in enumerate(dl):
                print("Beginning vocabulary build: " + f)
                path_in      = os.path.join(inDir,f)
                
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

                # read in dataframe 
                df = pd.read_csv(path_in, names = ['tweets'])

                # basic preprocessing
                mask = df['tweets'].str.len() > 10 
                df   = df.loc[mask]
                df   = cudf.from_pandas(df)
                df   = basic_clean(df)

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
                gc.collect()

                # add vocabulary from current file to CountVectorizer vocabulary
                countvec.partial_fit(df['tweets'])
                print("End " + f)

                # count rows of data
                num_data_rows += len(df.index)
                print(num_data_rows)
                print(len(df.index))
        else:
            countvec.vocabulary_ = countvec.vocabulary
            vocab = len(countvec.vocabulary_)
        
        # Save fitted CountVectorizer and vocabulary
        pickle.dump(countvec, open(os.path.join(save_dir, COUNTVECTOR_FILEPATH), 'wb'))
        vocab = len(countvec.vocabulary_)
        df_voc = cudf.DataFrame({'words':countvec.vocabulary_})
        df_voc.to_csv(save_dir + "/" + VOCAB_FILEPATH)
        print("right after countvec partial fit vocab\n\n\n: ", vocab)

        # make directories to save:
        #  - X matrices
        #  - corpus (only needed if computing coherence)
        x_mat_dir = os.path.join(save_dir, X_MAT_FILEPATH_PREFIX)
        if not os.path.exists(x_mat_dir):
            os.makedirs(x_mat_dir)
        corpus_dir = os.path.join(save_dir, CORPUS_FILEPATH_PREFIX)
        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)
      

        # transform X matrices with fitted CountVectorizer and save to disk
        transform_time = 0.0
        if save_files == 1:
            for f in dl:
                print("Beginning CountVectorizer transform: " + f)
                path_in  = os.path.join(inDir,f)

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()

                # read in dataframe 
                df = pd.read_csv(path_in, names = ['tweets'])
             
                # basic preprocessing
                mask = df['tweets'].str.len() > 10 
                df   = df.loc[mask]
                df   = cudf.from_pandas(df)
                df   = basic_clean(df)

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                gc.collect()

                # transform data from current file
                t1 = time.time()
                corpus = countvec.transform(df['tweets'])
                t2 = time.time()
                transform_time += t2 - t1
                X_batch = tl.tensor(corpus.toarray()) 
                
                # save current X matrix and corpus to disk
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

        gc.collect()
    
    print("Transform Time:" + str(transform_time))


    # initialize TLDA using parameters from above
    tlda = TLDA(
        num_tops, alpha_0, n_iter_train, n_iter_test,learning_rate, 
        pca_batch_size = batch_size_pca, third_order_cumulant_batch = batch_size_grad, 
        gamma_shape = 1.0, smoothing = smoothing, theta=theta_param, ortho_loss_criterion = ortho_loss_param, random_seed = seed, 
        n_eigenvec = n_eigenvec,
    )

    tot_tlda_time = 0.0
    if stgd == 1:
        # keep track of iterations
        i = 0
        
        t1 = time.time()
        for f in dl:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

            print("Beginning TLDA: " + f)

            # load saved X matrix batch from disk
            X_batch = pickle.load(
                        open(save_dir + X_MAT_FILEPATH_PREFIX + Path(f).stem + '.obj','rb')
                    )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            gc.collect()


            t3 = time.time()
            # fit tensor LDA fully online
            if initialize_first_docs and i == 0:
                # fully fit tensor LDA on first batch
                tlda.fit(X_batch)
            else:
                # partial fit tensor LDA on remaining batches
                tlda.partial_fit_online(X_batch)

        t4 = time.time()
        print("New fit time" + str(t4-t3))
        tot_tlda_time += t4-t3
    
        del X_batch
        gc.collect()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()            
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
    
        i += 1
    else:
        tlda = pickle.load(open(exp_save_dir + TLDA_FILEPATH,'rb'))


    # save top words in each topic
    if recover_top_words == 1:
        n_top_words = 100  

        top_words_df = cudf.DataFrame({})
        for k in range(0,num_tops): 
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


def main():
    curr_dir = "metoo_evaluation_initialized_paper_exps/"

    # set parameters
    num_tops = 10
    alpha_0  = 0.01
    lr = 0.0001
    pca_dim = 40
    
    # run method to fit topics and save top words in each topic
    fit_topics(
        num_tops = num_tops,
        curr_dir = curr_dir,
        alpha_0 = alpha_0,
        learning_rate = lr,
        n_eigenvec = pca_dim
    )

if __name__ == "__main__":
    main()

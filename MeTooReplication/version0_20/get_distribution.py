import numpy as np
import math
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA

import tensorly as tl
from tensorly.cp_tensor import cp_mode_dot
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from pca import PCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal


from tlda_final import TLDA
import cumulant_gradient
import tensor_lda_util as tl_util
## Break down into steps, then re-engineer.

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
porter = PorterStemmer()


stop_words = (stopwords.words('english'))
added_words = ["amendment","family","get","adam","hear","feder","de","la","los","democrat","republican",
               'el', 'para', 'en', 'que',"lo",
               "amend","back","protect","commun","service","work","around","alway","november","august","january",
               "happen","ive","hall","nation","work","service","this","discuss","community","learn","congressional","amendment","speaker","say",
               "said","talk","congrats","pelosi","gop","congratulations","are","as","i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
               "you", "your", "yours","he","her","him","she","hers","that","be","with","their","they're","is","was","been","not","they","it","have",
               "will","has","by","for","madam","Speaker","Mister","Gentleman","Gentlewoman","lady","voinovich","kayla","111th","115th","114th","rodgers",      
               "clerk" ,    "honor" ,   "address"   ,     
               "house" , "start"   ,"amend","bipartisan","bill",   "114th"    ,   "congress"  ,     
               "one",   "thing"    ,"bring","put", "north","give","keep","pa","even","texa","year","join","well",
               "call",  "learned"    ,   "legislator","things" ,"things","can't","can","cant","will","go","going","let",
               "lets","let's","say","says","know","talk","talked","talks","lady","honorable","dont","think","said","something",
               "something","wont","people","make","want","went","goes","congressmen","people","person","like","come","from",
               "need","us"]

stop_words= list(np.append(stop_words,added_words))



class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [porter.stem(self.wnl.lemmatize(t,get_wordnet_pos(t))) for t in word_tokenize(articles)]
    
class StemTokenizer(object):
    def __init__(self):
        self.porter = PorterStemmer()
    def __call__(self, articles):
        return [self.porter.stem(t) for t in word_tokenize(articles)]




def get_distribution(data_filename="../data/TwitterSpeech.csv", total_tweets=300000, n_topic=20, alpha_0=0.003, n_iter_train=1000, n_iter_test=150, batch_size=30000, learning_rate=0.01):

    df = pd.read_csv(data_filename)
    

    countvec = CountVectorizer(tokenizer=StemTokenizer(),
                                    strip_accents = 'unicode', # works 
                                    stop_words = stop_words, # works
                                    lowercase = True, # works
                                    ngram_range = (1,2),
                                    max_df = 0.4, # works
                                    min_df = int(0.002*total_tweets))


    vectorized = countvec.fit_transform(df.tweet[ df.year>=2019][:total_tweets])


    M1 = np.mean(vectorized, axis=0)
    centered = vectorized - M1

    pca = PCA(n_topic, alpha_0, batch_size)
    pca.fit(centered) # fits PCA to  data, gives W
    whitened = pca.transform(centered) # produces a whitened words counts <W,x> for centered data x


    t = TLDA(n_topic, alpha_0=alpha_0, n_iter_train=n_iter_train, n_iter_test=n_iter_test, batch_size=batch_size,
         learning_rate=learning_rate)

    print('whitened shape:', whitened.shape)
    t.fit(whitened, verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD


    t.factors_ = pca.reverse_transform(t.factors_)

    t.predict(centered,w_mat=False,doc_predict=False)


    argsorted_factors = [ np.argsort(t.factors_[:,n]).tolist() for n in range(n_topic)]


    return argsorted_factors



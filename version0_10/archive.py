###############
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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






def get_distribution(data_filename="../data/TwitterSpeech.csv", total_tweets=300000, n_topic=20, alpha_0=0.003):


    df = pd.read_csv(data_filename)
    

    countvec = CountVectorizer(tokenizer=StemTokenizer(),
                                    strip_accents = 'unicode', # works 
                                    stop_words = stop_words, # works
                                    lowercase = True, # works
                                    ngram_range = (1,2),
                                    max_df = 0.4, # works
                                    min_df = int(0.002*total_tweets))


    vectorized = countvec.fit_transform(df.tweet[ df.year>=2019][:total_tweets])


    weights, factor = fit(vectorized, n_topic, alpha_0)
    doc_topic_dist, topic_word_dist = inference(vectorized, factor, weights)

    argsorted_factors = [ np.argsort(topic_word_dist[:,n]).tolist() for n in range(n_topic)]

    return argsorted_factors





    # Create Synthetic Data
print("Create Synthetic Data")
x, phi, _, alpha_0  = get_phi(num_tops, vocab, num_tweets, density)

print("Compute M1")
M1 = get_M1(x)

print("Compute M2")
M2 = get_M2(x, M1, alpha_0)

print("Decompose M2")
W, W_inv    = whiten(M2, num_tops) # W (n_words x n_topics)
X_whitened  = tl.dot(x, W)     # this returns the whitened counts in  (n_topics x n_docs)
M1_whitened = tl.dot(M1, W)   

=batched_tensor_dot(batch_tensor_dot(W.T,M2),W)


print("Fit LDA to get phi")
lambda_fit, phi     = fit(x, num_tops, alpha_0)

print("Acquire mu_hat")
doc_topic_dist, topic_word_dist = inference(x, phi, lambda_fit)


def test_fit():
    num_tops = 3
    num_tweets = 10
    density = 10
    vocab = 200

    x, phi, _, alpha_0 = test_util.get_phi(num_tops, vocab, num_tweets, density)
    weights, factors = tlc.fit(x, num_tops)
    _, RMSE = test_util.validate_gammad(factor, tl.tensor(phi), num_tops = num_tops)
    assert(RMSE.item() < 0.15)

def test_inference():
    num_tops = 3
    num_tweets = 10
    density = 10
    vocab = 200

    x, _, theta, alpha_0 = test_util.get_phi(num_tops, vocab, num_tweets, density)
    weights, factors = tlc.fit(x, num_tops, alpha_0)
    doc_topic_dist, topic_word_dist = tlc.inference(x, factor, weights)
    _, RMSE = test_util.validate_gammad(doc_topic_dist, theta, transpose=True, num_tops = num_tops)
    assert(RMSE.item() < 0.3)



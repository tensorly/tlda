import numpy as np 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Import TensorLy
import tensorly as tl

# Import functions from tensor lda method
from tlda_wrapper import TLDA
#from version0_10.final_demo import generate_top_words

tl.set_backend("numpy")
np.random.seed(0)

print("Loading dataset with 2 topics: Autos and Baseball")
# Fetch data from 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball']
newsgroups_test = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), 
                                     categories=categories)
texts = newsgroups_test.data

# Generate count vectors from documents.
vectorizer = CountVectorizer(min_df = 0.1, 
                             max_df = 0.4,
                      #       ngram_range = 1,
                             stop_words = stop_words)
vectors = vectorizer.fit_transform(texts).toarray()
vocab = vectorizer.get_feature_names()
print("Done loading dataset")

k = len(categories)

tlda = TLDA(
      n_topic = k, alpha_0 = 0.01, n_iter_train = 2000, n_iter_test = 10,
      learning_rate = 1e-5, pca_batch_size = 10000, 
      third_order_cumulant_batch = 10, theta=5.005, ortho_loss_criterion = 1, 
      random_seed = 0, n_eigenvec = k*5
    )

tlda.fit(vectors)
tlda.transform(X = vectors[:2], predict = True)
print(tlda.unwhitened_factors)

print("Done fitting tensor LDA")


import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from examples.demo_util import generate_top_words

# Import TensorLy
import tensorly as tl

# Import functions from tensor lda method
from tlda.tlda_wrapper import TLDA

seed = 0
tl.set_backend("numpy")
np.random.seed(seed)

print("Loading dataset with 2 topics: Autos and Baseball")
# Fetch data from 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball']
newsgroups_test = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
texts = newsgroups_test.data

# Generate count vectors from documents.
vectorizer = CountVectorizer(min_df = 0.05,
                             max_df = 0.2,
                             ngram_range = [1, 2],
                             stop_words = stop_words)
vectors = vectorizer.fit_transform(texts).toarray()
vocab = vectorizer.get_feature_names()

print("Running tensor LDA")
# Initialize Tensor LDA
k = len(categories)
tlda = TLDA(
      n_topic = k, alpha_0 = 0.01, n_iter_train = 2000, n_iter_test = 10,
      learning_rate = 1e-5, pca_batch_size = 10000,
      third_order_cumulant_batch = 10, theta=5.005, ortho_loss_criterion = 1,
      random_seed = seed, n_eigenvec = k*5
    )

# Fit Tensor LDA
tlda.fit(vectors)

print("Creating image to display fitted topics")
# Generate a wordcloud from the topics
generate_top_words(
    tlda.unwhitened_factors.T,
    vocab,
    np.argsort(tlda.weights_),
    k, 25)

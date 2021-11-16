import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle

import cupy as cp
import cudf
from cudf import Series
from cuml.feature_extraction.text import CountVectorizer

def generate_top_words(topic_word_dist, words, num_tops, top_n):
    '''save top words in each topic to a wordcloud'''
    cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=top_n,
                  colormap='tab10')

    #int(num_tops/3) + 1, 3
    fig, axes = plt.subplots(int(num_tops/2), 2, figsize=(50,50), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        if i < num_tops:
            cloud.generate_from_frequencies(dict(zip(words.to_pandas(), 
                                            cp.asnumpy(topic_word_dist[:,i]) )))
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig("../plots/wordcloud_tensor"+str(num_tops)+".png")
    return

factors_unwhitened = pickle.load(open('../data/learned_factors_MeToo.obj', 'rb'))
countvec           = pickle.load(open('../data/countvec.obj','rb'))
words = countvec.vocabulary_
num_tops = 30
top_n    = 30



generate_top_words(factors_unwhitened,words,num_tops,top_n)
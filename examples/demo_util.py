from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def generate_top_words(topic_word_dist, words, order, num_tops, top_n):
    '''helper function for visualizing top words in a wordcloud'''
    cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=top_n,
                  colormap='tab10')

    fig, axes = plt.subplots(1, 2, figsize=(7, 7),
                             sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        if i < num_tops:
            cloud.generate_from_frequencies(dict(zip(words, topic_word_dist[order[i], :])))
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(order[i]), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    return
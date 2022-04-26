import numpy as np

from get_distribution import get_distribution


def get_similarity(run1, run2, top_words=10, tolerance=5, n_topics=20):
    similarity = sum([any([len(set(list(run1[i])[-top_words:]) - set(list(run2[j])[-top_words:])) <= tolerance for i in range(n_topics)]) for j in range(n_topics)])
    print(np.array([[len(set(list(run1[i])[-top_words:]) - set(list(run2[j])[-top_words:])) for i in range(n_topics)] for j in range(n_topics)]))
    return similarity


run1 = get_distribution()
run2 = get_distribution()


print(get_similarity(run1, run2))
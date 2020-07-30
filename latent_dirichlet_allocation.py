import pickle

from gensim import corpora, models

from get_naver_news_query_results import *

with open('data/naver_news.pickle', 'rb') as f:
    corpus = pickle.load(f)
    f.close()

tokens_matrix = [[t[0] for t in corp['tokens'] if t[1] in tok.include_poses] for corp in corpus]
dictionary_LDA = corpora.Dictionary(tokens_matrix)
dictionary_LDA.filter_extremes(no_below=3)
bow_matrix = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in tokens_matrix]

num_topics = 10
lda_model = models.LdaModel(bow_matrix, num_topics=num_topics, \
                            id2word=dictionary_LDA, \
                            passes=4, alpha=[0.01] * num_topics, \
                            eta=[0.01] * len(dictionary_LDA.keys()))

for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(i) + ": " + topic)
    print()

print()

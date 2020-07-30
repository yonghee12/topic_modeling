import numpy as np
import pandas as pd

from get_naver_news_query_results import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from sklearn.decomposition import TruncatedSVD


def get_uniques_from_nested_lists(nested_lists):
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx(items, unique=False):
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else set(items)
    for idx, item in enumerate(items_unique):
        item2idx[item] = idx
        idx2item[idx] = item
    return item2idx, idx2item


corpus = []
queries = ['코로나', '정치', '경제', '인국공', '아이유', '설빙']
for query in queries:
    corpus += get_query_results(query, tokenize=True)
with open('data/naver_news.pickle', 'wb') as f:
    pickle.dump(corpus, f)
    f.close()

with open('data/naver_news.pickle', 'rb') as f:
    temp = pickle.load(f)
    f.close()

tokens_matrix = [[t[0] for t in corp['tokens'] if t[1] in tok.include_poses] for corp in corpus]
unique_tokens = get_uniques_from_nested_lists(tokens_matrix)
token2idx, idx2token = get_item2idx(unique_tokens)
token_length = len(token2idx)
dtm = [[0 for _ in range(token_length)] for _ in range(len(corpus))]
for row_idx, tokens in enumerate(tokens_matrix):
    for token in tokens:
        token_idx = token2idx[token]
        dtm[row_idx][token_idx] = 1

dtm_np = np.array(dtm)
dtm_df = pd.DataFrame(dtm_np, columns=token2idx.keys())

U, s, VT = np.linalg.svd(dtm_np, full_matrices=True)
U.round(2)
S = np.zeros(dtm_np.shape)
S[:len(s), :len(s)] = np.diag(s)

t = 10
S_t = S[:t, :t]
U_t = U[:, :t]
VT_t = VT[:t, :]

for vt_row in VT_t:
    top_words_idx = sorted(enumerate(vt_row), key=lambda x: x[1], reverse=True)[:20]
    print([(idx2token[i], round(s, 4)) for i, s in top_words_idx])


# using package
print()
tokens_df = pd.DataFrame(tokens_matrix)
tokens_matrix_joined = [' '.join(tokens) for tokens in tokens_matrix]
tfidfvectorizer = TfidfVectorizer(max_features=1000, max_df=0.5, smooth_idf=True)
countvectorizer = CountVectorizer()
X_count = countvectorizer.fit_transform(tokens_matrix_joined)
X_tfidf = tfidfvectorizer.fit_transform(tokens_matrix_joined)

svd_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X_count)
len(svd_model.components_)

terms = countvectorizer.get_feature_names()


def get_topics(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx + 1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])


get_topics(svd_model.components_, terms)

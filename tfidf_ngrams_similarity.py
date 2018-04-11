import re
from collections import defaultdict
from scipy import sparse

import preprocessing.clean_events as clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


events = defaultdict(list)

clean.clean_events_texts('tables_as_csv/events_meetup.csv', 'tables_as_csv/cleaned_events.csv')

# pd.set_option('display.max_colwidth', -1)
events = pd.read_csv('tables_as_csv/cleaned_events.csv')
print('The shape: %d x %d' % events.shape)
# print(events.head())

event_names = events['event_name']
vectorizer = TfidfVectorizer(analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(event_names)

import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

import time

t1 = time.time()
# matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 100, 0.8)

matches = cosine_similarity(tf_idf_matrix, None)
matches = sparse.csr_matrix(matches)
t = time.time() - t1
print("SELFTIMED:", t)


def get_matched_events(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top < sparsecols.size:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_id = np.empty([nr_matches], dtype=object)
    right_id = np.empty([nr_matches], dtype=object)
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_id[index] = name_vector['event_id'][sparserows[index]]
        right_id[index] = name_vector['event_id'][sparsecols[index]]
        left_side[index] = name_vector['event_name'][sparserows[index]]
        right_side[index] = name_vector['event_name'][sparsecols[index]]
        similarity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_id': left_id,
                         'left_side': left_side,
                         'right_id': right_id,
                         'right_side': right_side,
                         'similarity': similarity})


t1 = time.time()

matches_df = get_matched_events(matches, events, 1000000)
t = time.time() - t1

print("get_matched_events:", t)

t1 = time.time()

matches_df = matches_df[
    matches_df['similarity'].between(0.10000, 0.99999)].sort_values(by='similarity', ascending=False)
matches_df.to_csv('tables_as_csv/matches_df.csv')

t = time.time() - t1
print("filter similarity:", t)

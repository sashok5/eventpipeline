import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import re


nltk.download('punkt')
nltk.download('wordnet')

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))


def ngrams(string, n=8):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


class SimilarEvents:

    def __init__(self, past_event_details, events_array):
        """:param past_event_details: It has to be dictionary with {event_name: event_description}
           :param events_array:  Events array which has to be list of dictionaries of min 
           [{event_name, event_description}]"""
        self.past_event_details = past_event_details
        self.events_array = list(events_array)       # events data frame

    def __add_events(self):

        event_names = [event['event_name'] for event in self.events_array]
        new_event = self.past_event_details['event_name']

        # event check below
        if new_event in event_names:
            self.events_array.remove(self.past_event_details)
            self.events_array.append(self.past_event_details)
            return self.events_array
        else:
            self.events_array.append(self.past_event_details)
            return self.events_array

    @staticmethod
    def cos_sim(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product      
        """
        # cosθ = a.b/ |a||b|   ===> cosine angle
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def run(self, identifier='None'):
        all_events = self.__add_events()

        # we shall check events relation based on their mapping.
        events = [event['event_description'] for event in all_events]
        results = []

        tfidf_vectorizer = TfidfVectorizer(analyzer=ngrams)
        punctuation_stemming = str.maketrans("", "", string.punctuation)
        modified_arr = [[porter.stem(i.lower()) for i in tokenize(d.translate(punctuation_stemming)) if
                         i.lower() not in stop_words] for d in events]

        modified_documents = [' '.join(i) for i in modified_arr]

        # mapping cosine logic here
        if identifier == 'cosine':
            tfidf_matrix = tfidf_vectorizer.fit_transform(modified_documents)
            length = len(events)
            cosine = cosine_similarity(tfidf_matrix[length - 1], tfidf_matrix)
            values = cosine.tolist()[0]
            values.pop()

            final_values = {}
            # Arranging values in ascending order:
            for idx, value in enumerate(values):
                if value in final_values:
                    final_values[value].append(idx)
                else:
                    final_values[value] = [idx]

            if len(final_values) > 0:
                sorted_values = sorted(list(final_values.keys()))
                final_indices = [{value: final_values[value]} for value in sorted_values]
                for each_index_list in final_indices:
                    for each_idx in list(each_index_list.values())[0]:
                        all_events[each_idx]['cosine_value'] = list(each_index_list.keys())[0]
                        results.append(all_events[each_idx])
            return {"past_event": self.past_event_details,
                    "suggested_events": results}

        # creating vectorizer
        features = tfidf_vectorizer.fit_transform(modified_documents).todense()

        # Analyzing Euclidean suggestions
        l = len(modified_documents) - 1

        euclidean_suggestions = {}
        for i in range(l):
            distance = euclidean_distances(features[-1], features[i])[0][0]
            if distance in euclidean_suggestions.keys():
                euclidean_suggestions[distance].append(i)
            else:
                euclidean_suggestions[distance] = [i]

        sorted_keys = sorted(list(euclidean_suggestions.keys()))
        euclideans = [{k: euclidean_suggestions[k]} for k in sorted_keys]

        for each_euclidean in euclideans:
            for each_idx in list(each_euclidean.values())[0]:
                all_events[each_idx]['euclidean_distance'] = list(each_euclidean.keys())[0]
                results.append(all_events[each_idx])

        return {"past_event": self.past_event_details,
                "suggested_events": results}


def event_recommender(event_id, identifier=None):
    future_events = pd.read_csv('tables_as_csv/events.csv')
    future_events = future_events[['event_name', 'event_description']]
    future_events = future_events.to_dict(orient='records')

    # testing below, read steps
    # 1) Formatting Data to pass in event, all_events

    past_events = pd.read_csv('tables_as_csv/past_events.csv')
    past_events = past_events[['event name', 'event description']]
    past_events.columns = ['event_name', 'event_description']
    past_events = past_events.to_dict(orient='records')
    # selecting some random past event
    past_event = past_events[int(event_id)]

    # 1) If identifier is cosine
    if identifier == 'cosine':
        events = SimilarEvents(past_event, future_events).run(identifier='cosine')
        return events

    # 2) Testing our object below
    events = SimilarEvents(past_event, future_events).run()
    return events

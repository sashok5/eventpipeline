import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))


class SimilarEvents:

    def __init__(self, event_details, events_array):
        """:param event_details: It has to be dictionary with {event_name: event_description}
           :param events_array:  Events array which has to be list of dictionaries of min 
           [{event_name, event_description}]"""
        self.event_details = event_details
        self.events_array = list(events_array)       # events data frame

    def __add_events(self):

        event_names = [event['event_name'] for event in self.events_array]
        new_event = self.event_details['event_name']

        # event check below
        if new_event in event_names:
            return self.events_array
        else:
            self.events_array.append(self.event_details)
            return self.events_array

    @staticmethod
    def lem_tokens(tokens):
        lemmer = nltk.stem.WordNetLemmatizer()
        return [lemmer.lemmatize(token) for token in tokens]

    def lem_normalize(self, text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return self.lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def vectorize(self, documents):
        TfidfVec = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(documents)
        return (tfidf * tfidf.T).toarray()

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

    def run(self):
        all_events = self.__add_events()

        # we shall check events relation based on their mapping.
        events = [event['event_description'] for event in all_events]

        # formatting data by removing stemming.
        punctuation_stemming = str.maketrans("", "", string.punctuation)
        modified_arr = [[porter.stem(i.lower()) for i in tokenize(d.translate(punctuation_stemming)) if
                         i.lower() not in stop_words] for d in events]
        modified_documents = [' '.join(i) for i in modified_arr]

        # creating vectorizer
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(modified_documents).todense()

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
        euclideans = [euclidean_suggestions[k] for k in sorted_keys]

        for each_euclidean in euclideans:
            for each_idx in each_euclidean:
                yield all_events[each_idx]


def event_recommender(event_id):
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

    # 2) Testing our object below
    events_gen = SimilarEvents(past_event, future_events).run()
    results = [event for event in events_gen]
    return results


import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from db_model import Event, EventGroup, EventSimilarity, EventCategory
from parse_text import parse_text

nltk.download('punkt')
nltk.download('wordnet')

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))


def ngrams(string, n=8):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


class Event_Similarity:

    def __init__(self, dataset, session, min_similarity, max_similarity):
        self.dataset = dataset
        self.session = session
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity

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

    def save(self):

        # we shall check events relation based on their mapping.
        events = [event.group_name + ' ' + event.shortname + ' ' + event.title + ' ' + event.desc for event in
                  self.dataset]

        vectorizer = TfidfVectorizer(analyzer='word', min_df=0.02, max_df=0.75)

        modified_arr = [tokenize(parse_text(d)) for d in events]

        modified_documents = [' '.join(i) for i in modified_arr]

        # mapping cosine logic here
        matrix = vectorizer.fit_transform(modified_documents)
        cosine = cosine_similarity(matrix)

        x = 0

        # process similarity of each event and save
        for values in cosine.tolist():
            y = 0
            for value in values:
                if self.min_similarity < value < self.max_similarity:
                    # skip the events with the same title
                    event1 = self.dataset[x]
                    event2 = self.dataset[y]
                    similarity = EventSimilarity(event_id_1=event1.event_id,
                                                 event_id_2=event2.event_id,
                                                 similarity=value)
                    self.session.add(similarity)
                y += 1
            x += 1
            self.session.commit()


def gen_similarities(db_session, min_similarity, max_similarity, event_id=None, ):
    dataset = db_session.query(Event.event_id,
                               Event.title,
                               Event.desc,
                               EventGroup.group_name,
                               EventCategory.shortname) \
        .join(EventGroup,
              Event.group_id == EventGroup.group_id) \
        .join(EventCategory, Event.category_id == EventCategory.category_id).all()

    # wipe previous similarities
    db_session.query(EventSimilarity).delete()
    db_session.commit()

    Event_Similarity(dataset, db_session, min_similarity, max_similarity).save()

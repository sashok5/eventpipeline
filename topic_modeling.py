import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from db_model import Event, EventGroup, EventTopic, EventTopicMapping
from parse_text import parse_text
nltk.download('punkt')
nltk.download('wordnet')

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))


class Topic_Modeling:

    def __init__(self, dataset, session, num_of_topics, num_of_words):
        self.dataset = dataset
        self.session = session
        self.num_of_topics = num_of_topics
        self.num_of_words = num_of_words

    def create_topics(self):

        events = [event.group_name + ' ' + event.title + ' ' + event.desc for event in self.dataset]

        vectorizer = CountVectorizer(max_df=0.70, min_df=0.01)

        modified_documents = [parse_text(event) for event in events]

        tf = vectorizer.fit_transform(modified_documents)

        vocab = vectorizer.get_feature_names()

        model = lda.LDA(n_topics=self.num_of_topics, n_iter=1000)
        model.fit(tf)  # model.fit_transform(X) is also available
        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = self.num_of_words
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            topic = EventTopic(i, ' '.join(topic_words))
            self.session.add(topic)

        self.session.commit()

        doc_topic = model.doc_topic_
        for i, event in enumerate(self.dataset):
            mapping = EventTopicMapping(doc_topic[i].argmax(), event.event_id)
            self.session.add(mapping)

        self.session.commit()

    def collab(self):
        return


def topic_modeling(db_session, num_of_topics, num_of_words):
    dataset = db_session.query(Event.event_id,
                               Event.title,
                               Event.desc,
                               EventGroup.group_name) \
        .join(EventGroup,
              Event.group_id == EventGroup.group_id).all()

    # wipe previous topics
    db_session.query(EventTopic).delete()
    db_session.query(EventTopicMapping).delete()
    db_session.commit()

    tm = Topic_Modeling(dataset, db_session, num_of_topics, num_of_words)

    tm.create_topics()

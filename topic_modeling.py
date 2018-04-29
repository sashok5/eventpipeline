import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from db_model import Event, EventGroup, Topic, EventTopic, User, EventCategory
from parse_text import parse_text
from gensim import corpora, models, similarities
import gensim
import os
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

tokenizer = nltk.RegexpTokenizer(r'\w+')

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Topic_Modeling:

    def __init__(self, session, dataset, num_of_topics, num_of_words):
        self.dataset = dataset
        self.session = session
        self.num_of_topics = num_of_topics
        self.num_of_words = num_of_words

    def practice(self):

        users_all = self.session.query(User)
        for u, user in enumerate(users_all):
            user.attending = user.events_attending(self.session, user.user_id)

        with open("user_ids.dat", "w") as users_list:
            for i, user in enumerate(users_all):
                users_list.write(str(i) + ' ' + str(user.user_id))
                users_list.write("\n")

        events_all = self.session.query(Event)

        with open("event_ids", "w") as event_list:
            for i, event in enumerate(events):
                event_list.write(str(i) + ' ' + str(event.event_id))
                event_list.write("\n")

        user_events = []

        with open("users.dat", "w") as users_file:

            for user in users:
                users_file.write(str(len(user.attending)))
                for i, event_id in enumerate(user.attending):
                    users_file.write(' ' + str(event_id))
                users_file.write("\n")

        with open("items.dat", "w") as events_file:

            for event_index, event in enumerate(events):
                e = []
                for user_index, user in enumerate(users):
                    if event.event_id in user.attending:
                        e.append(user_index)
                events_file.write(str(len(e)))
                for i in e:
                    events_file.write(' ' + str(i))
                events_file.write("\n")

        '''
        # get the users and attending events
        with open("test.txt", "w") as users_file:
            users = self.session.query(User)
            for user in users:
                attending = user.events_attending(self.session, user.user_id)
                users_file.write(attending.size)
                for event_id in attending:
                    users_file.write(' ' + str(event_id))
                users_file.write("\n")
        '''

        # events = [event.group_name + ' ' + event.title + ' ' + event.desc for event in self.dataset]

        vectorizer = CountVectorizer(max_df=0.70, min_df=0.01)

        modified_documents = [parse_text(event) for event in events]

        tf = vectorizer.fit_transform(modified_documents)
        with open("terms.txt", "w") as terms_file:
            for word in vectorizer.get_feature_names():
                terms_file.write(word)
                terms_file.write("\n")

        with open("mult.dat", "w") as mult:
            for i, event in enumerate(events):
                mult.write(str(tf[i].size))
                for j in range(tf[i].size):
                    mult.write(' ' + str(tf[i].indices[j]) + ':' + str(tf[i].data[j]))
                mult.write("\n")

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
            topic = Topic(i, ' '.join(topic_words))
            self.session.add(topic)

        self.session.commit()

        doc_topic = model.doc_topic_
        for i, event in enumerate(self.dataset):
            mapping = EventTopic(doc_topic[i].argmax(), event.event_id)
            self.session.add(mapping)

        self.session.commit()

    def gensim_create_topics(self):

        data = [event.group_name + ' ' + event.title + ' ' + event.desc for event in self.dataset]

        events = [tokenizer.tokenize(parse_text(event)) for event in data]

        dictionary = corpora.Dictionary(events)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(event) for event in events]

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word=dictionary, passes=50)

        topics = ldamodel.print_topics(num_topics=6, num_words=10)

        for i, t in enumerate(topics):
            topic = Topic(i, t)
            print(ldamodel.get_topic_terms(i))
            self.session.add(topic)

        self.session.commit()

        for i, event in enumerate(self.dataset):
            for index, score in sorted(ldamodel[corpus[i]], key=lambda tup: -1 * tup[1]):
                event_topic = EventTopic(index, event.event_id, np.round(score, decimals=5))
                self.session.add(event_topic)

        self.session.commit()
        '''
        
        for i in range(3):
            for index, score in sorted(ldamodel[corpus[i]], key=lambda tup: -1 * tup[1]):
                print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 10)))
        '''

    def gensim_string_similarity(self, search_string):
        '''
        Using LSI to compare the search string to the documents and show best 20 results
        :param search_string:
        :return:
        '''
        # load the event names in the same sorted order as the dictionary
        events = self.session.query(Event.event_id,
                                    Event.title
                                    ).order_by('event_id').all()

        # create the dictionary if file does not exist
        if os.path.isfile('tmp/events.dict') is False:
            self.gensim_generate_corpus()

        dictionary = corpora.Dictionary.load('tmp/events.dict')
        corpus = corpora.MmCorpus('tmp/events.mm')

        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
        # let's find our search string
        vec_bow = dictionary.doc2bow(search_string.lower().split())

        vec_lsi = lsi[vec_bow]  # convert the query to LSI space

        index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

        sims = index[vec_lsi]  # perform a similarity query against the corpus

        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        # let's get the documents the search results are associated with
        results = []
        for i, s in enumerate(sims):
            results.append([events[s[0]].event_id, events[s[0]].title, s[1]])
            if i > 20:
                break

        df = pd.DataFrame(columns=('event_id', 'title', 'similarity'), data=results)

        return df.to_json(orient="records")

    def gensim_generate_corpus(self):
        data = self.session.query(Event.event_id,
                                  Event.title,
                                  Event.desc,
                                  EventGroup.group_name,
                                  EventCategory.shortname) \
            .join(EventGroup,
                  Event.group_id == EventGroup.group_id) \
            .join(EventCategory,
                  Event.category_id == EventCategory.category_id).order_by('event_id').all()

        prep_data = [event.group_name + ' ' + event.shortname + ' ' + event.title + ' ' + event.desc for event in
                     data]

        events = [tokenizer.tokenize(parse_text(event)) for event in prep_data]

        dictionary = corpora.Dictionary(events)
        dictionary.save('tmp/events.dict')

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(event) for event in events]
        corpora.MmCorpus.serialize('tmp/events.mm', corpus)


def practice(db_session, num_of_topics, num_of_words):
    dataset = db_session.query(Event.event_id,
                               Event.title,
                               Event.desc,
                               EventGroup.group_name) \
        .join(EventGroup,
              Event.group_id == EventGroup.group_id).all()

    tm = Topic_Modeling(dataset, db_session, num_of_topics, num_of_words)

    tm.practice()


def search_events(db_session, search_string):
    tm = Topic_Modeling(db_session, None, None, None)
    return tm.gensim_string_similarity(search_string)


def topic_modeling(db_session, num_of_topics, num_of_words, method="gensim"):
    dataset = db_session.query(Event.event_id,
                               Event.title,
                               Event.desc,
                               EventGroup.group_name) \
        .join(EventGroup,
              Event.group_id == EventGroup.group_id).all()

    # wipe previous topics
    db_session.query(Topic).delete()
    db_session.query(EventTopic).delete()
    db_session.commit()

    tm = Topic_Modeling(db_session, dataset, num_of_topics, num_of_words)

    if method == "gensim":
        tm.gensim_create_topics()
    else:
        tm.create_topics()


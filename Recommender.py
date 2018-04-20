import os
from datetime import date
from sqlalchemy import create_engine, func
from db_model import make_session, metadata
from migrate_data import migrate_dataset, create_events_additional, migrate_justevents
import event_similarity
import topic_modeling

class Recommender(object):

    def __init__(self, db_uri, events_file_path):
        self.engine = create_engine(db_uri)
        self.events_file_path = events_file_path
        self.session = make_session(self.engine)
        metadata.create_all(bind=self.engine)

    def full_migration(self):
        # 1. Save all data to tables
        migrate_dataset(self.events_file_path, self.session)

    def migate_justevents(self):
        migrate_justevents(self.events_file_path, self.session)

    def cosine_similarity(self):
        """
        Train the recommender by creating a tfidf vectorizer and find cosine similarity between events
        :return:
        """
        min_similarity = 0.1
        max_similarity = 0.89
        event_similarity.gen_similarities(self.session, min_similarity, max_similarity)

    def generate_topics(self):
        """
        Generate topics and map the topics to events using LDA
        """
        # 1. Create topics using LDA model
        num_of_topics = 100
        num_of_words = 10
        topic_modeling.topic_modeling(self.session, num_of_topics, num_of_words)

    def add_attending(self, user_id, event_id):
        # 1. assign topic / create new one

        # 2.
        return


db_connection_string = 'postgresql://admin:admin@localhost:5432/groupup'
csv_files = 'tables_as_csv/'
recommender = Recommender(db_connection_string, csv_files)

#recommender.full_migration()
#recommender.cosine_similarity()
#recommender.generate_topics()
#recommender.migrate_additional_events()
#recommender.migate_justevents()

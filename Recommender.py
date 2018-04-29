import os
from datetime import date
from sqlalchemy import create_engine, func
from db_model import make_session, metadata, User, Recommended, Popular, Attendances, EventTopic
from migrate_data import migrate_dataset, create_events_additional, migrate_justevents
import event_similarity
import topic_modeling


class Recommender(object):

    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        self.events_file_path = 'tables_as_csv/'
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

    def update_user_topics(self):
        topics = self.session.query(Even)
        users = db_session.query(User)

    def search_keyword(self, search_string):
        return topic_modeling.search_events(self.session, search_string)

    def generate_popular(self):
        # Clear the current popular events
        self.session.query(Popular).delete()

        popular_events = Popular.get_popular_events(self.session)
        for i, element in enumerate(popular_events):
            e = Popular(element[0].event_id, element[0].attendance_count, element[1].click_count)
            self.session.add(e)
        self.session.commit()

    def recommend_events(self, user_id):
        '''
        The logic should be that if user is following at least one person then recommend the events that the person is attending

        :param user_id:
        :return:
        '''
        # Delete previously recommended events for this user
        self.session.query(Recommended).filter(Recommended.user_id == user_id).delete()
        self.session.commit()

        # We are going to limit number of events to 20 and rank them based on the below algorithm
        rank = 20
        following_user_events = User.following_user_events(self.session, user_id)
        for i, event in enumerate(following_user_events):
            recommendation = Recommended(event.event_id, user_id, rank)
            self.session.add(recommendation)
            rank = rank - 1
        self.session.commit()
        # Recommend person preferences for the user based on the topics user likes

        # Remaining events will be from Popular
        # Skip events that were already recommended higher
        popular_events = self.session.query(Popular).order_by(Popular.num_of_attending, Popular.num_of_clicks)
        for i, event in enumerate(popular_events):
            existing = self.session.query(Recommended).filter(Recommended.event_id == event.event_id).first()
            if existing is None:
                recommendation = Recommended(event.event_id, user_id, rank)
                self.session.add(recommendation)
            else:
                existing.rank = existing.rank + rank
            rank = rank - 1
        self.session.commit()

    def practice(self):
        num_of_topics = 100
        num_of_words = 10
        topic_modeling.practice(self.session, num_of_topics, num_of_words)


db_connection_string = 'postgresql://admin:admin@localhost:5432/groupup'

recommender = Recommender(db_connection_string)

# recommender.full_migration()
# recommender.cosine_similarity()
# recommender.generate_topics()
# recommender.migrate_additional_events()
# recommender.migate_justevents()

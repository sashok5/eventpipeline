import enum
import os
from datetime import date
from sqlalchemy import create_engine, func
from db_model import make_session, metadata, User, Recommended, Popular, Event, EventTopic, UserTopic, Attendances
import event_similarity
import topic_modeling
import collab_filtering
import numpy as np
from config import db_connection_string


class Recommender(object):

    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        self.session = make_session(self.engine)
        metadata.create_all(bind=self.engine)

    def cosine_similarity(self):
        """
        Train the recommender by creating a tfidf vectorizer and find cosine similarity between events
        :return:
        """
        min_similarity = 0.1
        max_similarity = 0.90
        event_similarity.gen_similarities(self.session, min_similarity, max_similarity)

    def generate_topics(self, num_topics, num_words, iterations):
        """
        Generate topics and map the topics to events using LDA
        """
        # 1. Create topics using LDA model
        topic_modeling.topic_modeling(self.session, num_topics, num_words, iterations, "lda")

    def update_user_topics(self):
        topic_modeling.update_user_topics(self.session)

    def run_collab_filtering(self):
        collab_filtering.Collab_Filtering(self.session).run()

    def search_keyword(self, search_string):
        return topic_modeling.search_events(self.session, search_string)

    def generate_popular(self):
        # Clear the current popular events
        self.session.query(Popular).delete()

        popular_events = Popular.get_popular_events(self.session)
        for i, element in enumerate(popular_events):
            e = Popular(element[0].event_id, element[0].attendance_count, 0)
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

        # for filtering future events user is already attending
        attending = self.session.query(Attendances.event_id).join(Event, Event.event_id == Attendances.event_id)\
            .filter(Attendances.user_id == user_id, Event.event_date >= func.now()).all()

        # We are going to limit number of events to 20 and rank them based on the below algorithm
        rank = 50

        # Recommend person preferences for the user based on the topics user likes
        user_topics = self.session.query(UserTopic).filter(UserTopic.user_id == user_id).order_by(
            'predicted_score DESC').all()

        for i, ut in enumerate(user_topics):
            future_recommended_events = self.session.query(Event).join(EventTopic,
                                                                       EventTopic.event_id == Event.event_id).filter(
                Event.event_date >= func.now(), EventTopic.topic_id == ut.topic_id).order_by(
                'event_topics.score DESC').limit(2)
            for j, fe in enumerate(future_recommended_events):
                if fe.event_id not in attending:
                    recommendation = Recommended(fe.event_id, user_id, rank, 4)
                    self.session.add(recommendation)
                    rank = rank - 1

        following_user_events = User.following_user_events(self.session, user_id)
        for i, event in enumerate(following_user_events):
            if event.event_id not in attending:
                recommendation = Recommended(event.event_id, user_id, rank, 2)
                self.session.add(recommendation)
                rank = rank - 1
        self.session.commit()

        # Remaining events will be from Popular
        # Skip events that were already recommended higher
        popular_events = self.session.query(Popular).order_by(Popular.num_of_attending, Popular.num_of_clicks)
        for i, event in enumerate(popular_events):
            existing = self.session.query(Recommended).filter(Recommended.event_id == event.event_id).first()
            if event.event_id not in attending:
                if existing is None:
                    recommendation = Recommended(event.event_id, user_id, rank, 1)
                    self.session.add(recommendation)
                else:
                    existing.rank = existing.rank + rank
                rank = rank - 1
        self.session.commit()


recommender = Recommender(db_connection_string)

# recommender.full_migration()
# recommender.cosine_similarity()
# recommender.generate_topics()
# recommender.migrate_additional_events()
# recommender.migate_justevents()
# recommender.run_collab_filtering()
# recommender.migate_justevents()
#recommender.migrate_users_only()
#recommender.randomize_attendances()


class RecommendationType(enum.Enum):
    popular = 1
    follower = 2
    content = 3
    collab = 4
    search = 5

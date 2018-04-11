import db_manager as db
from db_manager import Popular, Events, Attendances, Relationships
from scipy.spatial.distance import correlation
import pandas as pd


def follower_data(user_id):

    users_following = db.DBSession.query(Relationships.c.follower_id,
                                         Relationships.c.followed_id,
                                         Events.c.event_id,
                                         Events.c.category_id,
                                         Popular.rank)\
            .join(Attendances, Attendances.c.user_id == Relationships.c.followed_id)\
            .join(Events, Events.c.event_id == Attendances.c.event_id)\
            .join(Popular, Popular.event_id == Events.c.event_id)\
            .filter(Relationships.c.follower_id == user_id)\

    return users_following


def recommend_follower_events(user_id):

    data = follower_data(user_id)
    print(data)

db.init_sqlalchemy()
recommend_follower_events(1)

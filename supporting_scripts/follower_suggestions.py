import db_manager as db
from db_manager import Popular, Events, Attendances, Relationships, Recommended
from scipy.spatial.distance import correlation
import pandas as pd


def clear_recommended(user_id):
    db.DBSession.query(Recommended) \
        .filter(Recommended.user_id == user_id).delete()


def follower_data(user_id):
    data = db.DBSession.query(Relationships.c.follower_id.label('user_id'),
                              Events.c.event_id,
                              (Popular.num_of_attending * Popular.num_of_clicks).label(
                                  'rank')) \
        .join(Attendances, Attendances.c.user_id == Relationships.c.followed_id) \
        .join(Events, Events.c.event_id == Attendances.c.event_id) \
        .join(Popular, Popular.event_id == Events.c.event_id) \
        .filter(Relationships.c.follower_id == user_id and Events.c.event_date >= db.func.now())

    return data


def gen_follower_events(user_id):
    db.init_sqlalchemy()

    clear_recommended(user_id)

    data = follower_data(user_id)

    result = []

    for n in data:
        mapping = Recommended(event_id=n.event_id, user_id=n.user_id, rank=n.rank)
        db.DBSession.add(mapping)
        result.append(mapping)

    db.DBSession.commit()

    return result



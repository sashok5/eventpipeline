import db_manager as db
from db_manager import Popular, Events, Attendances, EventViews


def clear_popular():
    db.DBSession.query(Popular).delete()


def future_events_by_popularity():
    events = db.DBSession.query(Events.c.event_id,
                                db.func.count(Attendances.c.event_id).label('c1'),
                                db.func.count(EventViews.c.event_id).label('c2')) \
        .filter(Events.c.event_date >= db.func.now()) \
        .outerjoin(Attendances, Events.c.event_id == Attendances.c.event_id) \
        .outerjoin(EventViews, Events.c.event_id == EventViews.c.event_id) \
        .group_by(Events.c.event_id)
    return events


def gen_popular():

    db.init_sqlalchemy()

    clear_popular()

    pop = future_events_by_popularity()

    result = []

    for n in pop:
        popular_event = Popular(event_id=n.event_id, num_of_attending=n.c1, num_of_clicks=n.c2)
        db.DBSession.add(popular_event)
        result.append(popular_event)

    db.DBSession.commit()

    return result

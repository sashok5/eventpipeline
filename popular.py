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


def save_popular():
    clear_popular()

    pop = future_events_by_popularity()

    for n in pop:
        popular_event = Popular(event_id=n.event_id, num_of_attending=n.c1, num_of_clicks=n.c2)
        db.DBSession.add(popular_event)

    db.DBSession.commit()


if __name__ == '__main__':
    db.init_sqlalchemy()
    save_popular()

import os
from tarfile import open as open_tar, TarError
from datetime import date
import pandas as pd
from db_model import User, Event, EventGroup, EventCategory, UserInGroup, Attendances, EventView
import numpy as np
import datetime as dt

from psycopg2.extensions import register_adapter, AsIs


def adapt_numpy_int64(numpy_int64):
    """ Adapting numpy.int64 type to SQL-conform int type using psycopg extension, see [1]_ for more info.
+
+    References
+    ----------
+    .. [1] http://initd.org/psycopg/docs/advanced.html#adapting-new-python-types-to-sql-syntax
+    """
    return AsIs(numpy_int64)


def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


register_adapter(np.int64, adapt_numpy_int64)
register_adapter(np.float32, adapt_numpy_float32)


def migrate_dataset(path_to_datasets, db_session):
    events_path = os.path.join(path_to_datasets, "events_meetup.csv")
    categories_path = os.path.join(path_to_datasets, "categories.csv")
    groups_path = os.path.join(path_to_datasets, "groups.csv")
    users_path = os.path.join(path_to_datasets, "members.csv")

    # clean db first
    db_session.query(EventView).delete()
    db_session.query(Attendances).delete()
    db_session.query(Event).delete()
    db_session.query(User).delete()
    db_session.query(UserInGroup).delete()
    db_session.query(EventGroup).delete()
    db_session.query(EventCategory).delete()

    # reset sequences
    db_session.execute("ALTER SEQUENCE users_user_id_seq RESTART WITH 1;")

    # save changes
    db_session.commit()

    # create categories
    create_categories(categories_path, db_session)

    # create groups
    create_groups(groups_path, db_session)
    db_session.commit()

    # create fake user #1
    User.create_user(db_session, 1, 'test', 'test@test.com', 'None', 'NY', '2007-02-14')
    db_session.commit()

    # create events
    create_events(events_path, db_session)

    create_events_additional('events_with_groups1.csv', db_session)
    create_events_additional('events_with_groups.csv', db_session)

    # create users
    create_users(users_path, db_session)
    db_session.commit()

    # generate some fake attendance
    for event in db_session.query(Event):
        Attendances.generate_attendances(db_session, event.event_id, event.group_id, event.rsvp_count)
    db_session.commit()

    db_session.execute("SELECT setval('users_user_id_seq', max(user_id)) FROM users;")
    db_session.commit()


def migrate_justevents(path, db_session):
    db_session.query(EventView).delete()
    db_session.query(Attendances).delete()
    db_session.query(Event).delete()
    # create events
    create_events(os.path.join(path, "events_meetup.csv"), db_session)
    create_events_additional('events_with_groups.csv', db_session)
    create_events_additional('events_with_groups1.csv', db_session)
    create_events_additional('events_with_groups2.csv', db_session)
    create_events_additional('events_with_groups3.csv', db_session)
    create_events_additional('events_with_groups4.csv', db_session)

    for event in db_session.query(Event):
        Attendances.generate_attendances(db_session, event.event_id, event.group_id, event.rsvp_count)
    db_session.commit()


def create_events(events_path, session):
    start = dt.datetime.now()
    chunksize = 1000
    j = 0
    index_start = 1

    for events_df in pd.read_csv(events_path, chunksize=chunksize, iterator=True, encoding='latin1'):
        events_df.index += index_start

        j += 1
        print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

        for index, r in events_df.iterrows():
            category_id = session.query(EventGroup.category_id).filter(EventGroup.group_id == r['group_id']).first()

            existing_event = session.query(Event.title).filter(Event.title == str(r['event_name'])).first()

            if not pd.isnull(r['event_name']) and not pd.isnull(r['description']) and existing_event is None:
                event = Event(title=r['event_name'], desc=r['description'], addr=r['venue.address_1'],
                              host=r['venue.name'], city=r['venue.city'],
                              state=r['venue.state'], zip=None, rsvp_count=r['yes_rsvp_count'],
                              group_id=r['group_id'], created=r['created'], event_date=r['event_time'],
                              category_id=category_id, meetup_id=r['event_id'])
                session.add(event)

        session.commit()
        index_start = events_df.index[-1] + 1


def create_categories(categories_path, session):
    try:
        categories_df = pd.read_csv(categories_path)
    except IOError as exc:
        raise IndexingError("Could not open categories file %s" & exc)

    for index, r in categories_df.iterrows():
        category = EventCategory(category_id=r["category_id"], name=r["category_name"], shortname=r["shortname"])
        session.add(category)

    session.commit()


def create_groups(groups_path, session):
    start = dt.datetime.now()
    chunksize = 1000
    j = 0
    index_start = 1

    for groups_df in pd.read_csv(groups_path, chunksize=chunksize, iterator=True, encoding='latin1'):
        groups_df.index += index_start

        j += 1
        print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

        for index, r in groups_df.iterrows():
            group = EventGroup(group_id=r["group_id"], category_id=r["category_id"], city=r["city"], state=r["state"],
                               country=r["country"], created=r["created"], group_name=r["group_name"],
                               description=r["description"], members=r["members"])
            session.add(group)

        session.commit()
        index_start = groups_df.index[-1] + 1


def create_users(users_path, session):
    start = dt.datetime.now()
    chunksize = 1000
    max = 100000
    j = 0
    index_start = 1

    for df in pd.read_csv(users_path, chunksize=chunksize, iterator=True, encoding='latin1'):

        df.index += index_start

        j += 1
        print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

        member_ids = df.member_id.unique()

        for member_id in member_ids:
            u = df[(df.member_id == member_id)].iloc[0]

            if not User.user_exists(session, member_id):
                user = User(
                    user_id=u["member_id"],
                    name=u["member_name"],
                    email="test" + str(member_id) + "@test.com",
                    city=u["city"],
                    state=u["state"],
                    created=u["joined"])
                session.add(user)

            for index, r in df[(df.member_id == member_id)].iterrows():
                ug = UserInGroup(u["member_id"], r["group_id"])
                session.add(ug)

            session.commit()

        index_start = df.index[-1] + 1
        if j * chunksize > max:
            break


def create_events_additional(filepath, session):
    start = dt.datetime.now()
    chunksize = 1000
    j = 0
    index_start = 1

    for events_df in pd.read_csv(filepath, chunksize=chunksize, iterator=True, encoding='latin1'):
        events_df.index += index_start

        j += 1
        print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

        for index, r in events_df.iterrows():
            existing_event = session.query(Event.title).filter(Event.title == r['name']).first()

            if not pd.isnull(r['name']) and not pd.isnull(r['description']) and existing_event is None:
                event = Event(title=r['name'], desc=r['description'], addr=r['venue.address_1'],
                              host=r['venue.name'], city=r['venue.city'],
                              state=r['venue.state'], zip=None, rsvp_count=r['yes_rsvp_count'],
                              group_id=r['group_id'], created=pd.to_datetime(r['created']),
                              event_date=pd.to_datetime(r['time']),
                              category_id=r['category_id'], meetup_id=r['id'])
                session.add(event)

        session.commit()
        index_start = events_df.index[-1] + 1


class IndexingError(Exception):
    """
    Exception raised when there was a problem while loading the Netflix data
    set into a database.

    """
    pass

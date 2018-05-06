import os
from datetime import date
import pandas as pd
from sqlalchemy import create_engine, func
from db_model import make_session, metadata, User, Event, EventGroup, EventCategory, UserInGroup, Attendances, EventView
import numpy as np
import datetime as dt
import random
from psycopg2.extensions import register_adapter, AsIs
from config import db_connection_string

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


class MeetupDataMigration(object):

    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        self.session = make_session(self.engine)
        metadata.create_all(bind=self.engine)
        self.path_to_datasets = 'tables_as_csv/'

    def migrate_dataset(self):
        events_path = os.path.join(self.path_to_datasets, "events_meetup.csv")
        categories_path = os.path.join(self.path_to_datasets, "categories.csv")
        groups_path = os.path.join(self.path_to_datasets, "groups.csv")
        users_path = os.path.join(self.path_to_datasets, "members.csv")

        # clean db first
        self.session.query(EventView).delete()
        self.session.query(Attendances).delete()
        self.session.query(Event).delete()
        self.session.query(User).delete()
        self.session.query(UserInGroup).delete()
        self.session.query(EventGroup).delete()
        self.session.query(EventCategory).delete()

        # reset sequences
        self.session.execute("ALTER SEQUENCE users_user_id_seq RESTART WITH 1;")

        # save changes
        self.session.commit()

        # create categories
        self.create_categories(categories_path)

        # create groups
        self.create_groups(groups_path)
        self.session.commit()

        # create fake user #1
        User.create_user(self.session, 1, 'test', 'test@test.com', 'None', 'NY', '2007-02-14')
        self.session.commit()

        # create events
        self.create_events(events_path)

        self.create_events_additional('events_with_groups1.csv')
        self.create_events_additional('events_with_groups.csv')

        # create users
        self.create_users(users_path)
        self.session.commit()

        # generate some fake attendance
        for event in self.session.query(Event):
            Attendances.generate_attendances(self.session, event.event_id, event.group_id, event.rsvp_count)
            self.session.commit()

            self.session.execute("SELECT setval('users_user_id_seq', max(user_id)) FROM users;")
            self.session.commit()

    def migrate_justevents(self, path):
        self.session.query(EventView).delete()
        self.session.query(Attendances).delete()
        self.session.query(Event).delete()
        # create events
        self.create_events(os.path.join(path, "events_meetup.csv"))
        self.create_events_additional(os.path.join(path, 'events_with_groups.csv'))
        self.create_events_additional(os.path.join(path, 'events_with_groups1.csv'))
        self.create_events_additional(os.path.join(path, 'events_with_groups2.csv'))
        self.create_events_additional(os.path.join(path, 'events_with_groups3.csv'))
        self.create_events_additional(os.path.join(path, 'events_with_groups4.csv'))
        self.create_events_additional(os.path.join(path, 'events_with_groups5.csv'))

        for event in self.session.query(Event):
            Attendances.generate_attendances(self.session, event.event_id, event.group_id, event.rsvp_count)
            self.session.commit()

    def create_events(self, events_path):
        start = dt.datetime.now()
        chunksize = 1000
        j = 0
        index_start = 1

        for events_df in pd.read_csv(events_path, chunksize=chunksize, iterator=True, encoding='latin1'):
            events_df.index += index_start

            j += 1
            print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

            for index, r in events_df.iterrows():
                category_id = self.session.query(EventGroup.category_id).filter(EventGroup.group_id == r['group_id']).first()

                existing_event = self.session.query(Event.title).filter(Event.title == str(r['event_name'])).first()

                if not pd.isnull(r['event_name']) and not pd.isnull(r['description']) and existing_event is None and len(
                        r['description']) > 50:
                    event = Event(title=r['event_name'], desc=r['description'], addr=r['venue.address_1'],
                                  host=r['venue.name'], city=r['venue.city'],
                                  state=r['venue.state'], zip=None, rsvp_count=r['yes_rsvp_count'],
                                  group_id=r['group_id'], created=r['created'], event_date=r['event_time'],
                                  category_id=category_id, meetup_id=r['event_id'])
                    self.session.add(event)
                self.session.commit()
            index_start = events_df.index[-1] + 1

    def create_categories(self, categories_path):
        try:
            categories_df = pd.read_csv(categories_path)
        except IOError as exc:
            raise IndexingError("Could not open categories file %s" & exc)

        for index, r in categories_df.iterrows():
            category = EventCategory(category_id=r["category_id"], name=r["category_name"], shortname=r["shortname"])
            self.session.add(category)

        self.session.commit()

    def create_groups(self, groups_path):
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
                self.session.add(group)

            self.session.commit()
            index_start = groups_df.index[-1] + 1

    def create_users(self, users_path):
        start = dt.datetime.now()
        chunksize = 1000
        max = 1000000
        j = 0
        index_start = 1

        for df in pd.read_csv(users_path, chunksize=chunksize, iterator=True, encoding='latin1'):

            df.index += index_start

            j += 1
            print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

            member_ids = df.member_id.unique()

            for member_id in member_ids:
                u = df[(df.member_id == member_id)].iloc[0]

                if not User.user_exists(self.session, member_id):
                    user = User(
                        user_id=u["member_id"],
                        name=u["member_name"],
                        email="test" + str(member_id) + "@test.com",
                        city=u["city"],
                        state=u["state"],
                        created=u["joined"])
                    user.password_digest = '$2a$10$Cz58MmQLYjIXeecXTdP8NO8ZuKldiUKbOxWUIlbKweazCg4EFwUtq'  # password = test
                    self.session.add(user)

                for index, r in df[(df.member_id == member_id)].iterrows():
                    ug = UserInGroup(u["member_id"], r["group_id"])
                    self.session.add(ug)

                self.session.commit()

            index_start = df.index[-1] + 1
            if j * chunksize > max:
                break

    def create_events_additional(self, filepath):
        start = dt.datetime.now()
        chunksize = 1000
        j = 0
        index_start = 1

        for events_df in pd.read_csv(filepath, chunksize=chunksize, iterator=True, encoding='latin1'):
            events_df.index += index_start

            j += 1
            print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize))

            for index, r in events_df.iterrows():
                existing_event = self.session.query(Event.title).filter(Event.title == r['name']).first()

                if not pd.isnull(r['name']) and not pd.isnull(r['description']) and existing_event is None and len(
                        r['description']) > 50:
                    event = Event(title=r['name'], desc=r['description'], addr=r['venue.address_1'],
                                  host=r['venue.name'], city=r['venue.city'],
                                  state=r['venue.state'], zip=None, rsvp_count=r['yes_rsvp_count'],
                                  group_id=r['group_id'], created=dt.datetime.fromtimestamp(r['created'] / 1e3),
                                  event_date=dt.datetime.fromtimestamp(r['time'] / 1e3),
                                  category_id=r['category_id'], meetup_id=r['id'])
                    self.session.add(event)

            self.session.commit()
            index_start = events_df.index[-1] + 1

    def randomize_attendances(self):
        # generate some fake attendance
        # first delete existing attendances
        self.session.query(Attendances).delete()
        self.session.commit()

        for event in self.session.query(Event):
            # get the group and pull random number of users associated with the group at least 5 and most 50
            num_attending = np.random.random_integers(5, 50)
            # get all users in the groups
            users = self.session.query(User.user_id).join(UserInGroup, UserInGroup.user_id == User.user_id).filter(
                UserInGroup.group_id == event.group_id).all()
            if num_attending > len(users):
                num_attending = len(users)
            random_users = random.sample(users, num_attending)
            for user_id in random_users:
                a = Attendances(user_id, event.event_id, 3)
                self.session.add(a)
        self.session.commit()

    def migrate_users_only(self):
        self.session.query(Attendances).delete()
        self.session.query(User).filter(User.user_id != 1).delete()
        self.session.query(UserInGroup).delete()
        # reset sequences
        self.session.execute("ALTER SEQUENCE users_user_id_seq RESTART WITH 2;")
        self.session.commit()

        # create users
        users_path = "tables_as_csv/members.csv"
        self.create_users(users_path, self.session)
        self.session.commit()

        self.session.execute("SELECT setval('users_user_id_seq', max(user_id)) FROM users;")
        self.session.commit()


class IndexingError(Exception):
    """
    Exception raised when there was a problem while loading the Netflix data
    set into a database.

    """
    pass


if (__name__ == '__main__'):
    migration = MeetupDataMigration(db_connection_string)
    # do whatever needs to be done

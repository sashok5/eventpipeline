from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, func
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

Base = declarative_base()
DBSession = scoped_session(sessionmaker())
metadata = MetaData()
engine = None


class Popular(Base):
    __tablename__ = 'popular'
    event_id = Column(Integer, primary_key=True)
    title = Column('title', String)
    rank = Column('rank', Integer)


def init_sqlalchemy(dbname='postgresql://admin:admin@localhost:5432/groupup'):
    global engine
    engine = create_engine(dbname, echo=False)
    DBSession.remove()
    DBSession.configure(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)


def clear_popular():

    DBSession.query(Popular).delete()


def pop_popular():
    init_sqlalchemy()
    clear_popular()

    events = Table('events', Base.metadata, autoload=True, autoload_with=engine)

    users = Table('users', Base.metadata, autoload=True, autoload_with=engine)

    attendances = Table('attendances', Base.metadata, autoload=True, autoload_with=engine)

    relationships = Table('relationships', Base.metadata, autoload=True, autoload_with=engine)

    ranked_events = DBSession.query(events.c.event_id, events.c.title, func.count(events.c.event_id).label('count')) \
        .filter(events.c.event_date >= func.now()) \
        .outerjoin(attendances, events.c.event_id == attendances.c.event_id) \
        .group_by(events.c.event_id, events.c.title) \
        .order_by(func.count(events.c.event_id).desc())

    for n in ranked_events:
        popular_event = Popular(event_id=n.event_id, title=n.title, rank=n.count)
        DBSession.add(popular_event)

    DBSession.commit()


if __name__ == '__main__':
    pop_popular()
